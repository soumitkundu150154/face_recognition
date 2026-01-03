import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import requests
from typing import List, Tuple

# facenet-pytorch
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# ==============================
# CONFIG
# ==============================
YOLO_MODEL_PATH = "yolov8n.pt"            # your YOLO model (person detection)
AUTHORIZED_PATH = "authorized_faces"     # folder with images named <person_name>.jpg
SNAP_DIR = "intruder_snaps"
LOG_FILE = "log.xlsx"
FACE_THRESHOLD = 0.8   # cosine similarity threshold (higher = stricter). Tune this.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Telegram (keep them secret in real projects)
TELEGRAM_TOKEN = "Your_bot_token" 
CHAT_ID = "Your_Chat_ID"

# Make sure required folders exist
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(AUTHORIZED_PATH, exist_ok=True)

# ==============================
# 1. LOAD YOLO (person detector)
# ==============================
print("[INFO] Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)

# ==============================
# 2. LOAD Face models: MTCNN + InceptionResnet
# ==============================
print(f"[INFO] Setting up face detector/recognizer on device={DEVICE} ...")
mtcnn = MTCNN(keep_all=True, device=DEVICE)  # detect faces
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)  # embeddings

# ==============================
# 3. HELPERS: embedding & similarity
# ==============================
def get_embedding_from_image(img_bgr: np.ndarray) -> np.ndarray:
    """
    Given an OpenCV BGR image, runs MTCNN to get aligned face tensor(s)
    and returns the first face's L2-normalized embedding as a numpy array.
    Returns None if no face found.
    """
    if img_bgr is None:
        return None

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # mtcnn(img) returns a torch.Tensor of shape (3,H,W) for single face
    # or (N,3,H,W) for multiple faces, or None if no face found.
    with torch.no_grad():
        face_tensors = mtcnn(img_rgb)  # <- correct public API

    if face_tensors is None:
        return None

    # Ensure batch dimension
    if isinstance(face_tensors, torch.Tensor):
        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)  # (1,3,H,W)
    else:
        return None

    # Use the first face tensor
    face_tensor = face_tensors[0].to(DEVICE)      # (3,H,W)
    face_tensor = face_tensor.unsqueeze(0)        # (1,3,H,W)

    with torch.no_grad():
        emb = resnet(face_tensor)                 # (1,512)

    emb = emb.cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    if norm == 0:
        return None
    emb = emb / norm
    return emb


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ==============================
# 4. LOAD AUTHORIZED FACES
# ==============================
authorized_embeddings: List[np.ndarray] = []
authorized_names: List[str] = []

print("[INFO] Loading authorized faces from:", AUTHORIZED_PATH)
for fname in os.listdir(AUTHORIZED_PATH):
    low = fname.lower()
    if not (low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png")):
        continue
    p = os.path.join(AUTHORIZED_PATH, fname)
    img = cv2.imread(p)
    if img is None:
        print("[WARN] Could not read", p)
        continue
    emb = get_embedding_from_image(img)
    if emb is None:
        print(f"[WARN] No face detected in authorized image {fname}, skipping.")
        continue
    authorized_embeddings.append(emb)
    authorized_names.append(os.path.splitext(fname)[0])

print("[INFO] Authorized loaded:", authorized_names)

# ==============================
# 5. TELEGRAM ALERT
# ==============================
def send_telegram_alert(image_path: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        with open(image_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": CHAT_ID, "caption": "🚨 INTRUDER ALERT! Unknown person detected."}
            resp = requests.post(url, files=files, data=data, timeout=10)
            if resp.status_code != 200:
                print("[WARN] Telegram API returned", resp.status_code, resp.text[:200])
    except Exception as e:
        print("[ERROR] Telegram exception:", e)

# ==============================
# 6. LOGGING
# ==============================
def log_intruder(name: str, image_path: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, name, image_path]], columns=["Time", "Person", "Image"])
    if not os.path.exists(LOG_FILE):
        df.to_excel(LOG_FILE, index=False)
    else:
        old = pd.read_excel(LOG_FILE)
        new = pd.concat([old, df], ignore_index=True)
        new.to_excel(LOG_FILE, index=False)

# ==============================
# 7. Recognition helper
# ==============================
def recognize_embedding(emb: np.ndarray, auth_embs: List[np.ndarray], auth_names: List[str], threshold: float) -> Tuple[str, float]:
    if len(auth_embs) == 0:
        return "Unknown", 0.0
    auth_arr = np.vstack(auth_embs)  # (N, dim)
    # cosine similarity with all
    sims = (auth_arr @ emb) / (np.linalg.norm(auth_arr, axis=1) * np.linalg.norm(emb) + 1e-10)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return auth_names[best_idx], best_sim
    else:
        return "Unknown", best_sim

# ==============================
# 8. MAIN LOOP: capture -> detect persons -> detect faces -> recognize
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam (0)")

print("[INFO] System started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO detection (using ultralytics model)
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                # clamp
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                # try detect face(s) inside the person ROI using MTCNN
                try:
                    # mtcnn.detect returns boxes in ROI coordinate space
                    boxes, probs = mtcnn.detect(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    print("[WARN] MTCNN detect error:", e)
                    boxes = None

                name = "Unknown"
                best_score = 0.0

                if boxes is not None and len(boxes) > 0:
                    # iterate over detected faces (choose the one with best similarity)
                    for i, b in enumerate(boxes):
                        bx = b.astype(int)
                        fx1, fy1, fx2, fy2 = bx[0], bx[1], bx[2], bx[3]
                        # clamp inside ROI
                        fx1, fy1 = max(0, fx1), max(0, fy1)
                        fx2, fy2 = min(person_roi.shape[1], fx2), min(person_roi.shape[0], fy2)
                        face_crop = person_roi[fy1:fy2, fx1:fx2]
                        if face_crop.size == 0:
                            continue
                        emb = get_embedding_from_image(face_crop)
                        if emb is None:
                            continue
                        cand_name, sim = recognize_embedding(emb, authorized_embeddings, authorized_names, FACE_THRESHOLD)
                        # keep best similarity match
                        if sim > best_score:
                            best_score = sim
                            name = cand_name

                # Draw rectangle and label on original frame
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label_text = name if name != "Unknown" else f"Unknown ({best_score:.2f})"
                cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # If unknown, save + send alert + log
                if name == "Unknown":
                    snap_path = os.path.join(SNAP_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snap_path, frame)
                    send_telegram_alert(snap_path)
                    log_intruder("Unknown", snap_path)
                    print("[ALERT] Intruder detected:", snap_path)

    cv2.imshow("AI Surveillance System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
