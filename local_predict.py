# ==========================================================
# 🔥 Xception + LSTM Deepfake Detector (VS CODE / LOCAL)
# ==========================================================

import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image

# --- CONFIGURATION (UPDATE THESE PATHS LOCALLY) ---
# IMPORTANT: Place your model file in the same directory as this script.
LOCAL_MODEL_PATH = 'xception_lstm_best.pth' 
SEQ_LEN = 10

# ==========================================================
# 1️⃣ Define model (same architecture used during training)
# ==========================================================
class XceptionLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("legacy_xception", pretrained=False, num_classes=0)
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, seq, c, h, w = x.size()
        x = x.view(b * seq, c, h, w)
        feats = self.backbone(x)
        feats = feats.view(b, seq, -1)
        _, (h_n, _) = self.lstm(feats)
        out = self.classifier(h_n[-1])
        return out.squeeze(1)

# ==========================================================
# 2️⃣ Load model checkpoint (CPU)
# ==========================================================
device = torch.device("cpu")

model = XceptionLSTMModel()
try:
    checkpoint = torch.load(
        LOCAL_MODEL_PATH,
        map_location=device,
        weights_only=False
    )
    # Handle the model loading from the checkpoint structure
    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on CPU from {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Could not load model from {LOCAL_MODEL_PATH}. Check path/file integrity.")
    print(f"Details: {e}")
    exit()

# ==========================================================
# 3️⃣ Helper functions
# ==========================================================
def crop_face(frame):
    # Dummy face crop (center crop for simplicity)
    h, w, _ = frame.shape
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    return frame[y1:y2, x1:x2]

def transform(img):
    """Resize, convert to tensor, normalize (0-1), and reorder channels (HWC -> CWH)."""
    img = cv2.resize(img, (299, 299))  # Xception input size
    # OpenCV uses BGR, model expects RGB. Convert BGR to RGB here for consistent transformation
    img = img[:, :, ::-1].copy() 
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    return img

def select_local_file():
    """Opens a file dialog to select a video or image file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select a Video or Image File to Predict",
        filetypes=[
            ("Media Files", "*.mp4 *.avi *.mov *.mkv *.jpg *.jpeg *.png"),
            ("All Files", "*.*")
        ]
    )
    return file_path

# ==========================================================
# 4️⃣ Prediction function (CPU)
# ==========================================================
def predict_path(path, seq_len=SEQ_LEN):
    if not path:
        print("No file selected.")
        return None
        
    ext = path.split('.')[-1].lower()
    frames = []

    if ext in ['mp4', 'avi', 'mov', 'mkv']:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print("❌ ERROR: Could not open video file.")
            return None
            
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < seq_len:
            print(f"⚠️ Video too short ({total} frames). Using all frames.")
            seq_len = total

        idxs = np.linspace(0, total - 1, seq_len, dtype=int)
        
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            face = crop_face(frame)
            frames.append(transform(face)) # transform expects BGR input from cv2.read()
        cap.release()

    elif ext in ['jpg', 'jpeg', 'png']:
        # Use OpenCV for image reading to maintain BGR consistency
        frame = cv2.imread(path)
        if frame is None:
             print("❌ ERROR: Could not read image file.")
             return None
             
        face = crop_face(frame)
        face_t = transform(face)
        frames = [face_t] * seq_len # Repeat frame for sequence model

    else:
        print("❌ Unsupported file type:", ext)
        return None

    if len(frames) == 0:
        print("⚠️ No valid frames found!")
        return None

    frames = torch.stack(frames).unsqueeze(0).to(device)  # (1, seq_len, C, H, W)

    with torch.no_grad():
        prob = float(model(frames).item())

    # Compute confidence (using original 0.55 threshold)
    if prob > 0.55:
        label = "FAKE"
        confidence = prob * 100
    else:
        label = "REAL"
        confidence = (1 - prob) * 100

    print(f"[{label}] Deepfake Probability: {prob:.4f} | Confidence: {confidence:.2f}%")
    return {"label": label, "probability": prob, "confidence": confidence}

# ==========================================================
# 5️⃣ File Select & Predict (LOCAL)
# ==========================================================
if __name__ == "__main__":
    print("\n--- Starting Local Deepfake Detector ---")
    
    file_path = select_local_file()

    if file_path:
        print("\n🔹 Selected File:", file_path)
        result = predict_path(file_path)
        if result:
            print(f"\n==============================================")
            print(f"✅ FINAL PREDICTION: {result['label']} ({result['confidence']:.2f}% confident)")
            print(f"==============================================")
    else:
        print("No file selected. Exiting.")