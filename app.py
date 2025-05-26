import cv2
import os
import time
import logging
import shutil
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
from werkzeug.utils import secure_filename
import requests
import json

# For image captioning (install transformers, torch, torchvision)
from transformers import pipeline

# Set up logging for debugging and performance monitoring
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------------------
# Panorama creation functions
# ----------------------------
def extract_frames_from_video(video_path, num_images, output_folder="extracted_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []
    
    if num_images > total_frames:
        num_images = total_frames
    
    interval = total_frames // num_images
    frame_count = 0
    extracted_count = 0
    image_paths = []
    
    while cap.isOpened() and extracted_count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % interval == 0:
            output_path = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(output_path, frame)
            image_paths.append(output_path)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    return image_paths

def create_panorama_from_video(video_path, num_images):
    image_paths = extract_frames_from_video(video_path, num_images, OUTPUT_FOLDER)
    if len(image_paths) < 2:
        return None
    
    images = [cv2.imread(path) for path in image_paths]
    if any(img is None for img in images):
        return None

    try:
        stitcher = cv2.Stitcher_create()
    except AttributeError:
        stitcher = cv2.createStitcher()
    
    status, panorama = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        return None
    else:
        output_path = os.path.join(OUTPUT_FOLDER, "panorama_output.jpg")
        cv2.imwrite(output_path, panorama)
        return output_path

# ----------------------------
# Helper functions for orientation and LLM inference
# ----------------------------
def get_direction(theta):
    """Determine direction based on azimuth angle in a 360-degree panorama."""
    if -45 <= theta <= 45:
        return "in front"
    elif 45 < theta <= 135:
        return "to the right"
    elif -135 <= theta < -45:
        return "to the left"
    else:
        return "behind"

def generate_description(prompt):
    """Generate a description using Gemma 3 12B via Ollama."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:12b",
        "prompt": prompt
    }
    response = requests.post(url, json=payload, stream=True)
    full_response = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            if "response" in data:
                full_response += data["response"]
            if data.get("done", False):
                break
    # Removing markdown symbols if any persist (e.g., **)
    return full_response.replace("**", "")

# ----------------------------
# Load image captioning model
# ----------------------------
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# ----------------------------
# Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    logging.info("Video uploaded. Starting panorama stitching...")
    
    num_images = 50
    panorama_path = create_panorama_from_video(video_path, num_images)
    
    if panorama_path:
        panorama_url = f"/static/output/{os.path.basename(panorama_path)}"
        logging.info("Panorama created successfully.")
        return jsonify({'panorama_url': panorama_url})
    else:
        logging.error("Failed to create panorama.")
        return jsonify({'error': 'Failed to create panorama'}), 500

@app.route('/object_detection', methods=['GET'])
def object_detection():
    panorama_path = os.path.join(app.config['OUTPUT_FOLDER'], "panorama_output.jpg")
    if not os.path.exists(panorama_path):
        return jsonify({'error': 'Panorama image not found'}), 404

    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_prediction, get_sliced_prediction
    except ImportError:
        logging.error("SAHI library is not installed.")
        return jsonify({'error': 'SAHI library is not installed'}), 500

    # Load baseline detection model (YOLOv8x for non-sliced detection)
    logging.info("Loading baseline detection model (yolov8x)...")
    detection_model_baseline = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path="yolov8x.pt",
        confidence_threshold=0.4,
        device="cuda:0",  # Change to "cpu" if needed
    )

    # Run baseline detection (without SAHI slicing)
    logging.info("Running baseline detection (without SAHI slicing)...")
    result_without = get_prediction(panorama_path, detection_model_baseline)
    result_without.export_visuals(export_dir=app.config['OUTPUT_FOLDER'], hide_conf=True)
    without_src = os.path.join(app.config['OUTPUT_FOLDER'], 'prediction_visual.png')
    without_dst = os.path.join(app.config['OUTPUT_FOLDER'], 'object_detection_without_sahi.jpg')
    shutil.move(without_src, without_dst)

    # Load fine-tuned model (best.pt for SAHI detection)
    logging.info("Loading fine-tuned model (best.pt) for SAHI detection...")
    detection_model_sahi = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path="best.pt",
        confidence_threshold=0.2,
        device="cuda:0",  # Change to "cpu" if needed
    )

    # Run sliced detection using SAHI
    logging.info("Running sliced detection using SAHI...")
    result_with = get_sliced_prediction(
        panorama_path,
        detection_model_sahi,
        slice_height=1500,
        slice_width=1500,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )
    result_with.export_visuals(export_dir=app.config['OUTPUT_FOLDER'], hide_conf=True)
    with_src = os.path.join(app.config['OUTPUT_FOLDER'], 'prediction_visual.png')
    with_dst = os.path.join(app.config['OUTPUT_FOLDER'], 'object_detection_with_sahi.jpg')
    shutil.move(with_src, with_dst)

    # Generate image caption for richer context
    caption_result = captioner(panorama_path)
    image_caption = caption_result[0]['generated_text']

    # Calculate object orientations using detected objects
    direction_groups = {
        "in front": [], "to the left": [], "to the right": [], "behind": []
    }
    panorama = cv2.imread(panorama_path)
    height, width, _ = panorama.shape
    for obj in result_with.object_prediction_list:
        bbox = obj.bbox
        x, y, w, h = bbox.to_xywh()
        center_x = x + (w / 2)
        theta = (center_x / width) * 360 - 180
        direction = get_direction(theta)
        direction_groups[direction].append(obj.category.name)

    prompt_parts = []
    for direction, objects in direction_groups.items():
        if objects:
            objects_str = ", ".join(objects)
            prompt_parts.append(f"{direction}: {objects_str}")
    scene_description = ". ".join(prompt_parts) + "."

    # Prompt A: Exit instructions for a blind person.
    # Prompt A: Exit instructions (existing prompt)
    prompt_exit = (
        f"Below is a general description of a room: {image_caption} "
        f"and a list of detected objects by orientation: {scene_description} "
        "Your task is to provide clear, step-by-step instructions in an imperative tone on how to exit the room. "
        "Use commands such as 'Turn right' or 'Proceed straight', and explain which objects (and their positions: in front, to the left, to the right, or behind) will be useful as landmarks for navigation. "
        "Avoid using terms that rely on visual perception such as 'observe', 'visible', or 'see'. "
        "Do not use first-person language. "
        "Please provide the response in plain text without any markdown formatting."
    )

    prompt_general = (
        f"Below is a detailed panoramic view of a room: {image_caption} "
        f"and a list of detected objects organized by orientation: {scene_description} "
        "Your task is to produce a structured, concise plain text description with the following sections:\n"
        "Front: Briefly describe the main objects located in the front (2-3 lines).\n"
        "Left: Briefly describe the main objects to the left (2-3 lines).\n"
        "Right: Briefly describe the main objects to the right (2-3 lines).\n"
        "Behind: Briefly describe the main objects behind (2-3 lines).\n"
        "General Room Atmosphere: Summarize the room’s ambiance, purpose, and layout in 2-3 lines.\n"
        "Keep the descriptions concise and to the point. Do not use first-person language. "
        "Return the response in plain text without any markdown formatting."
    )


    # Generate LLM responses using Gemma 3 12B
    logging.info("Generating exit instructions with Gemma 3 12B...")
    description_exit = generate_description(prompt_exit)
    logging.info("Generating general room description with Gemma 3 12B...")
    description_general = generate_description(prompt_general)

    logging.info("Object detection and description generation completed.")
    return jsonify({
        'object_detection_without_sahi_url': f"/static/output/object_detection_without_sahi.jpg",
        'object_detection_with_sahi_url': f"/static/output/object_detection_with_sahi.jpg",
        'description_exit': description_exit,
        'description_general': description_general
    })

@app.route('/static/output/<filename>')
def serve_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
