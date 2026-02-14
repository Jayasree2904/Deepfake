# Deepfake Detection
This project implements a sophisticated Deepfake Detection system that combines the spatial feature extraction power of Xception with the temporal analysis capabilities of Long Short-Term Memory (LSTM) networks. It can analyze images and videos to determine if a human face has been manipulated.

This project supports:

📷 Webcam-based detection
🎥 Video-based deepfake classification
🖼️ Image-based prediction


# 📌 Dataset
Trained on:Celeb-DF-v2

## Tech Stack

**Backend:**

- Flask – Python web framework for API handling and server-side logic
- PyTorch – Deep learning framework for model training and inference
- timm – Pretrained Xception backbone integration
- OpenCV – Face detection and video frame processing
- NumPy – Numerical operations and frame sampling

**Frontend:**

- HTML – Structure of the web interface
- CSS – Styling and UI design
- JavaScript – Client-side interaction and prediction requests

**Model & Computer Vision:**

- Xception (CNN) – Spatial feature extraction
- LSTM – Temporal sequence modeling
- Haar Cascade – Face detection
- Celeb-DF-v2 – Training dataset


## Run Locally

Clone the project

```bash
  git clone https://github.com/Jayasree2904/Deepfake.git
```

Go to the project directory

```bash
  cd deepfake_app
```

Set Up Python Environment

```bash
  python -m venv venv
  venv\Scripts\activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the Flask Server

```bash
  python app.py
```

