document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const uploadSection = document.getElementById('upload-section');
    const spinner = document.getElementById('spinner');
    const outputImg = document.getElementById('panorama-output');
    const detectionSection = document.getElementById('detection-section');
    const detectionWithout = document.getElementById('detection-without');
    const detectionWith = document.getElementById('detection-with');

    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(uploadForm);

        spinner.style.display = 'block';
        spinner.querySelector('p').innerText = "Extracting frames, stitching frames...";
        uploadSection.style.display = 'none';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.error) {
                alert(result.error);
                spinner.style.display = 'none';
                uploadSection.style.display = 'block';
            } else {
                // Show panorama
                outputImg.src = result.panorama_url;
                outputImg.style.display = 'block';
                spinner.style.display = 'none';

                // Wait 5 seconds before removing panorama
                setTimeout(async () => {
                    outputImg.style.display = 'none';
                    spinner.style.display = 'block';
                    spinner.querySelector('p').innerText = "Object detection...";

                    const detectionResponse = await fetch('/object_detection');
                    const detectionResult = await detectionResponse.json();

                    spinner.style.display = 'none';

                    if (detectionResult.error) {
                        alert(detectionResult.error);
                    } else {
                        detectionWithout.src = detectionResult.object_detection_without_sahi_url;
                        detectionWith.src = detectionResult.object_detection_with_sahi_url;
                        detectionSection.style.display = 'block';
                    }
                }, 5000);
            }
        } catch (error) {
            alert('An error occurred: ' + error.message);
            spinner.style.display = 'none';
            uploadSection.style.display = 'block';
        }
    });
});
