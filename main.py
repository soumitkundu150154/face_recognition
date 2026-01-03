# import cv2
# import os
# import face_recognition 
# import numpy as np
# from ultralytics import YOLO
# import pandas as pd
# from datetime import datetime
# import requests
# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase
# from email import encoders


# # ==============================
# # 1. LOAD YOLO MODEL
# # ==============================
# model = YOLO("yolov8n.pt")   # small & fast model


# # ==============================
# # 2. LOAD AUTHORIZED FACES
# # ==============================
# authorized_path = "authorized_faces"
# authorized_encodings = []
# authorized_names = []

# for img_name in os.listdir(authorized_path):
#     img = face_recognition.load_image_file(os.path.join(authorized_path, img_name))
#     enc = face_recognition.face_encodings(img)

#     if len(enc) > 0:
#         authorized_encodings.append(enc[0])
#         authorized_names.append(img_name.split(".")[0])

# print("[INFO] Authorized persons loaded:", authorized_names)


# # ==============================
# # 3. TELEGRAM ALERT SYSTEM
# # ==============================
# TELEGRAM_TOKEN = "7223753160:AAG5vJGFf00GvULe4VNUwl3vUnTnUtnTpzA"
# CHAT_ID = "1147795607"

# def send_telegram_alert(image_path):
#     url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
#     files = {'photo': open(image_path, 'rb')}
#     data = {'chat_id': CHAT_ID, 'caption': "🚨 *INTRUDER ALERT!* Unknown person detected."}
#     requests.post(url, files=files, data=data)


# # # ==============================
# # # 4. EMAIL ALERT SYSTEM
# # # ==============================
# # EMAIL_SENDER = "suvendu.mondal@dsec.ac.in"
# # EMAIL_PASS = "your_app_password"
# # EMAIL_RECEIVER = "receiver@gmail.com"

# # def send_email_alert(image_path):
# #     subject = "INTRUDER ALERT!"
# #     body = "Unknown person detected. See attached snapshot."

# #     msg = MIMEMultipart()
# #     msg['From'] = EMAIL_SENDER
# #     msg['To'] = EMAIL_RECEIVER
# #     msg['Subject'] = subject

# #     msg.attach(MIMEText(body, 'plain'))

# #     # Attach Image
# #     attachment = open(image_path, 'rb')
# #     part = MIMEBase('application', 'octet-stream')
# #     part.set_payload(attachment.read())
# #     encoders.encode_base64(part)
# #     part.add_header('Content-Disposition', f"attachment; filename= {image_path}")
# #     msg.attach(part)

# #     server = smtplib.SMTP('smtp.gmail.com', 587)
# #     server.starttls()
# #     server.login(EMAIL_SENDER, EMAIL_PASS)
# #     server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
# #     server.quit()


# # # ==============================
# # 5. EXCEL LOGGING
# # ==============================
# def log_intruder(name, image_path):
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     df = pd.DataFrame([[timestamp, name, image_path]],
#                        columns=["Time", "Person", "Image"])

#     if not os.path.exists("log.xlsx"):
#         df.to_excel("log.xlsx", index=False)
#     else:
#         old = pd.read_excel("log.xlsx")
#         new = pd.concat([old, df], ignore_index=True)
#         new.to_excel("log.xlsx", index=False)


# # ==============================
# # 6. VIDEO CAPTURE
# # ==============================
# cap = cv2.VideoCapture(0)

# print("[INFO] System Started. Press Q to exit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame, stream=True)

#     for r in results:
#         for box in r.boxes:
#             cls = int(box.cls[0])
#             label = model.names[cls]

#             if label == "person":     # detect only humans
#                 x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
#                 person_roi = frame[y1:y2, x1:x2]

#                 # Face recognition
#                 face_locs = face_recognition.face_locations(person_roi)
#                 face_encs = face_recognition.face_encodings(person_roi, face_locs)

#                 name = "Unknown"

#                 for face_encoding in face_encs:
#                     matches = face_recognition.compare_faces(authorized_encodings, face_encoding)
#                     face_distances = face_recognition.face_distance(authorized_encodings, face_encoding)

#                     if len(face_distances) > 0:
#                         best_match = np.argmin(face_distances)
#                         if matches[best_match]:
#                             name = authorized_names[best_match]

#                 color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
#                 cv2.putText(frame, name, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#                 # If intruder detected – send alerts once
#                 if name == "Unknown":
#                     snap_path = f"intruder_snaps/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
#                     cv2.imwrite(snap_path, frame)

#                     send_telegram_alert(snap_path)
#                     #send_email_alert(snap_path)
#                     log_intruder("Unknown", snap_path)
#                     print("[ALERT] Intruder detected!")

#     cv2.imshow("AI Surveillance System", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()



import os
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import requests

# InsightFace imports
from insightface.app import FaceAnalysis

# ==============================
# CONFIG
# ==============================
YOLO_MODEL_PATH = "yolov8n.pt"
AUTHORIZED_PATH = "authorized_faces"
SNAP_DIR = "intruder_snaps"
LOG_FILE = "log.xlsx"
FACE_THRESHOLD = 1.0   # Euclidean distance threshold (tune: lower = stricter)

# Telegram config (you already have these values; keep them secret in real projects)
TELEGRAM_TOKEN = "7223753160:AAG5vJGFf00GvULe4VNUwl3vUnTnUtnTpzA"
CHAT_ID = "1147795607"

# Create required directories
os.makedirs(SNAP_DIR, exist_ok=True)
os.makedirs(AUTHORIZED_PATH, exist_ok=True)


# ==============================
# 1. LOAD YOLO MODEL
# ==============================
print("[INFO] Loading YOLO model...")
model = YOLO(YOLO_MODEL_PATH)   # small & fast model


# ==============================
# 2. SETUP INSIGHTFACE (detection+recognition)
# ==============================
print("[INFO] Initializing InsightFace...")
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])

# Try prepare with GPU (0) first, fallback to CPU (-1)
try:
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("[INFO] InsightFace prepared with ctx_id=0 (may use GPU).")
except Exception:
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("[INFO] InsightFace prepared with ctx_id=-1 (CPU).")


# ==============================
# 3. LOAD AUTHORIZED FACES (compute embeddings)
# ==============================
authorized_encodings = []
authorized_names = []

print("[INFO] Loading authorized faces from:", AUTHORIZED_PATH)
for img_name in os.listdir(AUTHORIZED_PATH):
    img_path = os.path.join(AUTHORIZED_PATH, img_name)
    if not (img_name.lower().endswith(".jpg") or img_name.lower().endswith(".png") or img_name.lower().endswith(".jpeg")):
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        continue

    faces = app.get(img)  # returns list of Face objects
    if len(faces) == 0:
        print(f"[WARN] No face found in {img_name}, skipping.")
        continue

    # Use first detected face's embedding
    emb = faces[0].embedding
    if emb is None:
        print(f"[WARN] No embedding for {img_name}, skipping.")
        continue

    authorized_encodings.append(emb)
    authorized_names.append(os.path.splitext(img_name)[0])

print("[INFO] Authorized persons loaded:", authorized_names)


# ==============================
# 4. TELEGRAM ALERT
# ==============================
def send_telegram_alert(image_path):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    try:
        files = {'photo': open(image_path, 'rb')}
        data = {
            'chat_id': CHAT_ID,
            'caption': "🚨 INTRUDER ALERT! Unknown person detected."
        }
        resp = requests.post(url, files=files, data=data, timeout=10)
        if resp.status_code != 200:
            print("[WARN] Telegram send failed:", resp.status_code, resp.text[:200])
    except Exception as e:
        print("[ERROR] Exception when sending telegram:", e)


# ==============================
# 5. EXCEL LOGGING
# ==============================
def log_intruder(name, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[timestamp, name, image_path]], columns=["Time", "Person", "Image"])

    if not os.path.exists(LOG_FILE):
        df.to_excel(LOG_FILE, index=False)
    else:
        old = pd.read_excel(LOG_FILE)
        new = pd.concat([old, df], ignore_index=True)
        new.to_excel(LOG_FILE, index=False)


# ==============================
# 6. HELPER: recognize face by embedding
# ==============================
def recognize_face(embedding, authorized_embeddings, authorized_names, threshold=FACE_THRESHOLD):
    if len(authorized_embeddings) == 0:
        return "Unknown", None

    # convert to numpy array and compute Euclidean distances
    auth_arr = np.vstack(authorized_embeddings)  # shape (N, dim)
    dists = np.linalg.norm(auth_arr - embedding, axis=1)
    best_idx = np.argmin(dists)
    best_dist = float(dists[best_idx])

    if best_dist <= threshold:
        return authorized_names[best_idx], best_dist
    else:
        return "Unknown", best_dist


# ==============================
# 7. VIDEO CAPTURE & MAIN LOOP
# ==============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam (index 0)")

print("[INFO] System Started. Press Q to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # run yolov8 detection on frame
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":     # detect only humans
                x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
                # clamp to image
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue

                # Use insightface to detect faces inside person_roi
                faces = app.get(person_roi)

                name = "Unknown"
                best_distance = None

                if len(faces) > 0:
                    # If multiple faces in ROI, pick the first with valid embedding that matches best
                    for face in faces:
                        emb = face.embedding
                        if emb is None:
                            continue
                        cand_name, cand_dist = recognize_face(emb, authorized_encodings, authorized_names)
                        # choose the recognized authorized face with smallest distance
                        if cand_name != "Unknown":
                            # recognized; take it
                            name = cand_name
                            best_distance = cand_dist
                            break
                        else:
                            # keep the smallest distance as fallback info
                            if best_distance is None or cand_dist < best_distance:
                                best_distance = cand_dist

                # Draw rectangle + label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label_text = name if name != "Unknown" else f"Unknown ({best_distance:.2f})" if best_distance is not None else "Unknown"
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # If intruder detected – send alerts once per detection
                if name == "Unknown":
                    snap_path = os.path.join(SNAP_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(snap_path, frame)

                    send_telegram_alert(snap_path)
                    log_intruder("Unknown", snap_path)
                    print("[ALERT] Intruder detected! Snapshot saved:", snap_path)

    cv2.imshow("AI Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
