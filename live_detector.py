# ==========================================================
# 🔥 Xception + LSTM Deepfake Detector (WITH FACE DETECTION)
# ==========================================================

import torch
import torch.nn as nn
import timm
import numpy as np
import cv2
import os
from collections import deque
from PIL import Image

# --- CONFIGURATION ---
LOCAL_MODEL_PATH = 'xception_lstm_best.pth' 
CAMERA_INDEX = 0 
SEQ_LEN = 10
FACE_CROP_SIZE = (299, 299) 
# ⚠️ UPDATE THIS PATH! Ensure the haarcascade file is present.
CASCADE_PATH = 'haarcascade_frontalface_default.xml' 

# ==========================================================
# 1️⃣ Define model (same architecture)
# ... (XceptionLSTMModel class remains unchanged) ...
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
# ... (Model loading section remains unchanged) ...
# ==========================================================
device = torch.device("cpu") 

model = XceptionLSTMModel()
try:
    checkpoint = torch.load(
        LOCAL_MODEL_PATH,
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False) 
    model.to(device)
    model.eval()
    print(f"✅ Model loaded on CPU from {LOCAL_MODEL_PATH}")
except Exception as e:
    print(f"❌ ERROR: Could not load model from {LOCAL_MODEL_PATH}. Check path/file integrity.")
    print(f"Details: {e}")
    exit()

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print(f"❌ ERROR: Failed to load Haar Cascade from {CASCADE_PATH}. Check file path.")
    exit()

# ==========================================================
# 3️⃣ Helper functions (Simplified)
# ==========================================================

def transform(img, target_size=(299, 299)):
    """Resize, convert to tensor, normalize (0-1), and reorder channels (HWC -> CWH)."""
    img = cv2.resize(img, target_size)
    img = img[:, :, ::-1].copy() 
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    return img

# ==========================================================
# 4️⃣ Single-Shot Detection Function (MAJOR UPDATE)
# ==========================================================
def single_shot_predict(model, face_cascade, frame_size=FACE_CROP_SIZE):
    """Captures ONE frame, detects face, predicts deepfake, and exits."""
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ ERROR: Cannot open webcam at index {CAMERA_INDEX}.")
        return

    # --- 1. Capture a single frame ---
    print("📷 Opening camera and capturing frame...")
    for _ in range(30): 
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read frame from camera.")
            cap.release()
            return
        
    cap.release() 
    print("✅ Camera closed.")

    frame = cv2.flip(frame, 1)
    frame_display = frame.copy()
    
    # --- 2. Face Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100) # Only look for reasonably large faces
    )

    if len(faces) == 0:
        # --- NO FACE DETECTED PATH ---
        label = "NO FACE DETECTED"
        color = (0, 165, 255) # Orange
        display_text = "NO FACE DETECTED"
        print(f"⚠️ {display_text} (Check if your face is visible in the frame.)")
        
        # Define a placeholder box for NO FACE
        h, w, _ = frame_display.shape
        x1, y1 = int(w * 0.2), int(h * 0.2)
        x2, y2 = int(w * 0.8), int(h * 0.8)
        
    else:
        # --- FACE DETECTED PATH (Prediction) ---
        (x, y, w_face, h_face) = faces[0] # Take the largest/first detected face
        
        # Extend the crop area slightly to capture context (optional, but good practice)
        x_pad = int(w_face * 0.15)
        y_pad = int(h_face * 0.15)
        
        # Define crop coordinates
        x1 = max(0, x - x_pad)
        y1 = max(0, y - y_pad)
        x2 = min(frame.shape[1], x + w_face + x_pad)
        y2 = min(frame.shape[0], y + h_face + y_pad)
        
        # Crop the face
        face_cropped = frame[y1:y2, x1:x2]
        
        # Preprocess
        face_tensor = transform(face_cropped, target_size=frame_size)
        input_sequence = face_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        # Prediction
        print("🧠 Running prediction...")
        with torch.no_grad():
            prob = model(input_sequence).item()

        # Interpret Result
        if prob > 0.5:
            label = "FAKE"
            conf = prob * 100
            color = (0, 0, 255) # Red
        else:
            label = "REAL"
            conf = (1 - prob) * 100
            color = (0, 255, 0) # Green
            
        display_text = f"RESULT: {label} | Conf: {conf:.2f}%"

    # --- 3. Display the result in a window ---
    print(f"\n✨ {display_text} ✨")
    
    # Draw the boundary box (uses x1, y1, x2, y2 from either NO FACE or FACE DETECTED path)
    cv2.rectangle(frame_display, (x1, y1), (x2, y2), color, 4)
    
    # Put the final text on the frame
    cv2.putText(frame_display, display_text, (10, frame_display.shape[0] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

    cv2.imshow('Deepfake Detector Result', frame_display)
    print("Press any key on the image window to close...")
    cv2.waitKey(0) 
    cv2.destroyAllWindows()


# ==========================================================
# 5️⃣ Execute the single-shot detection
# ==========================================================
if __name__ == "__main__":
    single_shot_predict(model, face_cascade)
    print("\nApplication finished.")