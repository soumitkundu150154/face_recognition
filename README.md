# Face Recognition & Alert System 🚨

## 📌 Overview
This project is a **Face Recognition and Alert System** designed to detect **unauthorized faces** in real time and immediately notify the user via **Telegram alerts**.

The system continuously monitors a video feed (webcam or camera module), compares detected faces against a set of **authorized faces**, and triggers an alert when an **unknown or unauthorized person** is identified.

This project can be used for:
- Home security
- Office or lab surveillance
- Restricted area monitoring
- Smart security systems

---

## ✨ Features
- 🔍 Real-time face detection and recognition  
- 👤 Authorized vs Unauthorized face classification  
- 📩 Instant **Telegram alert** on unauthorized detection  
- 📷 Supports webcam / external camera input  
- ⚡ Fast and lightweight processing  
- 🔒 Improves security with automated monitoring  

---

## 🛠️ Tech Stack
- **Python**
- **OpenCV**
- **Face Recognition Library**
- **Telegram Bot API**
- **NumPy**

---

## 📂 Project Structure
Face_Recognition
|
|- authorized_faces
|- intruder_snaps
|- main.py
|- new.py
|- test.py
|- yolov8n.pt


## Install dependencies 

pip install -r requirements.txt

## Telegram Bot Setup 
1. create a Telegram bot using BotFather
2. Copy the Bot Token
3. Get your Chat ID
4. Add the token and chat ID inside new.py


Examples 

TELEGRAM_TOKEN = "Your_bot_token"
CHAT_ID = "Your_Chat_ID"


# How to RUN

python new.py


Once running:

The camera will start detecting faces
Authorized faces will be ignored
Unauthorized faces will trigger a Telegram alert instantly


## 🚨 Alert System

When an unauthorized face is detected:

📸 Face snapshot is captured
📩 Telegram message is sent
⏱️ Alert is sent in real time


## 📸 Sample Use Case

“An unknown person enters a restricted room →
The system detects the face →
Sends an alert on Telegram →
User is notified instantly.”

## 🔮 Future Enhancements

Cloud-based face database
Mobile app integration
Multiple camera support
Alert throttling & logs
Face mask / emotion detection


## 👨‍💻 Author

Soumit Kundu
B.Tech CSE | AI, ROBOTICS & Computer Vision Enthusiast

# ⭐ If you like this project, don’t forget to star the repository!
