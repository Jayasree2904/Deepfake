document.addEventListener('DOMContentLoaded', () => {
    const uploadSection = document.getElementById('upload-section');
    const webcamSection = document.getElementById('webcam-section');
    const showUploadBtn = document.getElementById('show-upload');
    const showWebcamBtn = document.getElementById('show-webcam');
    const uploadForm = document.getElementById('upload-form');
    const captureButton = document.getElementById('capture-button');
    const videoElement = document.getElementById('webcam-video');
    const canvasElement = document.getElementById('webcam-canvas');
    const resultBox = document.getElementById('result-box');
    const resultLabel = document.getElementById('result-label');
    const resultConf = document.getElementById('result-conf');
    const loadingSpinner = document.getElementById('loading-spinner');
    const errorMsg = document.getElementById('error-message');

    let stream = null;

    // --- Utility Functions ---
    function showLoading() {
        loadingSpinner.classList.remove('hidden');
        resultBox.classList.add('hidden');
        errorMsg.classList.add('hidden');
    }

    function hideLoading() {
        loadingSpinner.classList.add('hidden');
    }

    function displayResult(data) {
        resultBox.classList.remove('hidden');
        
        resultLabel.className = ''; // Clear previous classes
        resultConf.textContent = '';
        const classLabel = data.label.toLowerCase().replace(/ /g, '-');
        if (data.label === "NO FACE DETECTED") {
            resultLabel.textContent = "NO FACE DETECTED";
            resultLabel.classList.add('result-noface');
        } else {
            resultLabel.textContent = `RESULT: ${data.label}`;
            resultLabel.classList.add(`result-${data.label.toLowerCase()}`);
            resultConf.textContent = `Confidence: ${data.confidence.toFixed(2)}%`;
        }
    }

    function displayError(message) {
        errorMsg.textContent = `Error: ${message}`;
        errorMsg.classList.remove('hidden');
        resultBox.classList.add('hidden');
    }

    // --- Webcam Toggle ---
    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            videoElement.srcObject = stream;
            videoElement.play();
        } catch (err) {
            displayError("Could not access webcam. Ensure camera is connected and permissions are granted.");
            console.error("Webcam error: ", err);
        }
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
    }

    // --- Mode Switching ---
    showUploadBtn.addEventListener('click', () => {
        uploadSection.classList.remove('hidden');
        webcamSection.classList.add('hidden');
        stopWebcam();
    });

    showWebcamBtn.addEventListener('click', () => {
        uploadSection.classList.add('hidden');
        webcamSection.classList.remove('hidden');
        startWebcam();
    });

    // --- File Upload Handler ---
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        showLoading();

        const formData = new FormData(uploadForm);
        
        try {
            const response = await fetch('/predict_upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            hideLoading();
            
            if (response.ok) {
                displayResult(data);
            } else {
                displayError(data.error || "File upload failed.");
            }
        } catch (error) {
            hideLoading();
            displayError("A network error occurred.");
            console.error("Upload error:", error);
        }
    });

    // --- Webcam Capture Handler ---
    captureButton.addEventListener('click', async () => {
        if (!stream) {
            displayError("Webcam not active. Cannot capture.");
            return;
        }

        showLoading();
        
        // Set canvas dimensions to the video dimensions
        const videoWidth = videoElement.videoWidth;
        const videoHeight = videoElement.videoHeight;
        canvasElement.width = videoWidth;
        canvasElement.height = videoHeight;

        // Draw the frame onto the canvas (mirroring is handled in CSS/video)
        const ctx = canvasElement.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, videoWidth, videoHeight);

        // Convert canvas content to base64 image data (JPEG for smaller size)
        const imageDataURL = canvasElement.toDataURL('image/jpeg', 0.8);

        try {
            const response = await fetch('/predict_webcam', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_data: imageDataURL })
            });

            const data = await response.json();
            hideLoading();

            if (response.ok) {
                displayResult(data);
            } else {
                displayError(data.error || "Webcam prediction failed.");
            }
        } catch (error) {
            hideLoading();
            displayError("A network error occurred during webcam prediction.");
            console.error("Webcam prediction error:", error);
        }
    });
});