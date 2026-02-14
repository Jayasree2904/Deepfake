# app.py

import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import os
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# --- CONFIGURATION ---
LOCAL_MODEL_PATH = 'xception_lstm_best.pth' 
CASCADE_PATH = 'haarcascade_frontalface_default.xml' 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov', 'mkv'}
SEQ_LEN = 10
IMG_SIZE = 299 # From training script
FACE_CROP_SIZE = (IMG_SIZE, IMG_SIZE)

# Normalization values used in your training script: T.Normalize([0.5]*3,[0.5]*3)
MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1) 
STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

# Recommended: Define a threshold margin to reduce false positives (adjust as needed)
FAKE_THRESHOLD = 0.5 # Prob >= 0.55 -> FAKE
REAL_THRESHOLD = 0.5 # Prob <= 0.45 -> REAL

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================================
# 1️⃣ Define model & Load Assets
# ==========================================================
# FIX 1: Model architecture now matches the training script exactly
class XceptionLSTMModel(nn.Module):
    def __init__(self, feat_dim=2048, hidden_dim=512, lstm_layers=1, dropout=0.5):
        super().__init__()
        
        # FIX 4: Use 'legacy_xception' to match the likely trained backbone and suppress warning
        self.backbone = timm.create_model('legacy_xception', pretrained=False, num_classes=0, global_pool='avg')
        self.feat_dim = feat_dim
        self.lstm = nn.LSTM(input_size=self.feat_dim, hidden_size=hidden_dim,
                            num_layers=lstm_layers, batch_first=True)
        
        # Classifier matches the training script (512 -> 256 -> Dropout -> 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, frames):
        b, seq_len, c, h, w = frames.size()
        frames = frames.view(b*seq_len,c,h,w)
        feats = self.backbone(frames)
        feats = feats.view(b,seq_len,-1)
        _,(h_n,_) = self.lstm(feats)
        final = h_n[-1]
        out = self.classifier(final)
        return out.squeeze(1)

# Load Model
device = torch.device("cpu")
model = XceptionLSTMModel() 

try:
    # 💥 CRITICAL FIX for "WeightsUnpickler error": Explicitly set weights_only=False
    # This allows loading older checkpoints that contain non-primitive Python objects (like numpy scalar).
    checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device, weights_only=False)
    
    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state, strict=False) 
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on CPU from {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model. Check path and model definition: {e}")
    exit()

# Load Haar Cascade
if not os.path.exists(CASCADE_PATH):
    print(f"❌ FATAL ERROR: Haar Cascade file not found at {CASCADE_PATH}")
    exit()
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"❌ FATAL ERROR: Failed to load Haar Cascade from {CASCADE_PATH}")
    exit()

# ==========================================================
# 2️⃣ Helper Functions (Face Crop, Transform, Prediction Logic)
# ==========================================================

# FIX 2: Correct transform function using [0.5, 0.5, 0.5] normalization
def transform(img, target_size=FACE_CROP_SIZE):
    """Resize, convert to tensor, and apply the correct training normalization."""
    img = cv2.resize(img, target_size)
    
    # 1. Convert BGR (OpenCV) to RGB (Model expectation)
    img = img[:, :, ::-1].copy() 
    
    # 2. Convert to Tensor and scale to [0, 1]
    img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    
    # 3. Apply Training Normalization (Mean=0.5, Std=0.5)
    img_tensor = (img_tensor - MEAN) / STD 
    
    return img_tensor

def detect_and_crop_face(frame):
    """Detects face using Haar Cascade, applies padding, and returns preprocessed tensor and box."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
    )

    if len(faces) == 0:
        return None, None
    
    # Take the largest/first detected face
    (x, y, w_face, h_face) = faces[0] 
    x_pad = int(w_face * 0.15)
    y_pad = int(h_face * 0.15)
    
    # Calculate crop coordinates
    x1 = max(0, x - x_pad)
    y1 = max(0, y - y_pad)
    x2 = min(frame.shape[1], x + w_face + x_pad)
    y2 = min(frame.shape[0], y + h_face + y_pad)
    
    face_cropped = frame[y1:y2, x1:x2]
    face_tensor = transform(face_cropped)
    
    # Return processed tensor and the box coordinates
    return face_tensor, [int(x1), int(y1), int(x2), int(y2)]

def predict_sequence(frames_tensors):
    """Runs prediction on a sequence of preprocessed face tensors."""
    if len(frames_tensors) == 0:
        return None

    # Stack tensors and create batch dimension: (1, SEQ_LEN, C, H, W)
    frames_sequence = torch.stack(frames_tensors).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        prob = float(model(frames_sequence).item())

    # Interpret Result with Confidence Margin
    if prob >= FAKE_THRESHOLD: 
        label = "FAKE"
        conf = prob * 100
        color_code = "red"
    elif prob <= REAL_THRESHOLD:
        label = "REAL"
        conf = (1 - prob) * 100
        color_code = "green"
    else:
        # Ambiguous result (too close to 50/50 split)
        label = "UNCLEAR"
        conf = max(prob, 1 - prob) * 100
        color_code = "orange" 
        
    return {
        "label": label, 
        "probability": prob,
        "confidence": float(conf), 
        "color": color_code
    }


# ==========================================================
# 3️⃣ Flask Routes
# ==========================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Renders the main upload/webcam page."""
    return render_template('index.html')

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    """Handles file uploads (image or video) and runs prediction on a sequence."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        ext = filename.rsplit('.', 1)[1].lower()
        face_tensors = []
        last_box = None
        
        # --- Image/Video Processing Logic ---
        if ext in ['jpg', 'jpeg', 'png']:
            # Handle Image File (Repeat the detected face SEQ_LEN times)
            frame = cv2.imread(filepath)
            if frame is not None:
                face_tensor, box = detect_and_crop_face(frame)
                if face_tensor is not None:
                    # FIX 3: Repeat the single processed frame for the sequence model
                    face_tensors = [face_tensor] * SEQ_LEN
                    last_box = box 

        elif ext in ['mp4', 'avi', 'mov', 'mkv']:
            # Handle Video File (Sample SEQ_LEN frames)
            cap = cv2.VideoCapture(filepath)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    # Sample frames linearly in time, like in the training script
                    idxs = np.linspace(0, total_frames - 1, SEQ_LEN, dtype=int)
                    
                    for i in idxs:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                        ret, frame = cap.read()
                        if ret:
                            face_tensor, box = detect_and_crop_face(frame)
                            if face_tensor is not None:
                                face_tensors.append(face_tensor)
                                last_box = box 
                cap.release()
        # ------------------------------------

        # Clean up the file after reading frame
        os.remove(filepath) 

        if len(face_tensors) < SEQ_LEN:
            return jsonify({
                "error": f"Failed to detect faces in {len(face_tensors)}/{SEQ_LEN} required frames.",
                "label": "ERROR",
                "confidence": 0.0
            }), 500

        # Run the sequence prediction
        result = predict_sequence(face_tensors)
        
        # Add the box to the result for visualization (using the last detected box)
        result['box'] = last_box
        return jsonify(result)
            
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/predict_webcam', methods=['POST'])
def predict_webcam():
    """Handles image data posted from the webcam (base64). Predicts using a repeated frame sequence."""
    try:
        data_url = request.json.get('image_data')
        if not data_url:
            return jsonify({"error": "No image data provided"}), 400
            
        # Strip the data:image/png;base64, part
        encoded_data = data_url.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 500
        
        # Detect and crop face
        face_tensor, box = detect_and_crop_face(frame)
        if face_tensor is None:
            return jsonify({"label": "NO FACE DETECTED", "probability": 0.0,"confidence": 0.0,"color": "grey", 
                "error": "No face detected in webcam image."}), 200
            
        # FIX 3: Repeat the single frame to create the sequence for the LSTM model
        face_tensors = [face_tensor] * SEQ_LEN
        
        # Run the sequence prediction
        result = predict_sequence(face_tensors)
        
        # Add the box to the result for visualization
        result['box'] = box
        return jsonify(result)

    except Exception as e:
        print(f"Webcam prediction error: {e}")
        return jsonify({"error": "Internal server error during processing"}), 500

if __name__ == '__main__':
    app.run(debug=True)