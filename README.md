Markdown
# Smart Attendance System with AI & IoT Embedded System

An advanced, real-time automated attendance system utilizing **Computer Vision (OpenCV)** and **Deep Learning (DeepFace)**[cite: 8, 11]. The system features asynchronous multi-threaded face tracking and recognition, designed to run smoothly on embedded platforms like the **NVIDIA Jetson Nano**[cite: 10, 11].

## 🚀 Key Features
- **Real-time Face Tracking:** Uses OpenCV Haar Cascades for fast, low-latency face detection[cite: 8, 11].
- **Asynchronous Deep Learning Inference:** Employs multi-threading to run heavy DeepFace facial recognition (ArcFace/FaceNet models) in the background, preventing camera feed lagging or flickering[cite: 11, 12].
- **Automated Dataset Collection:** Quick script to register new students by capturing multiple angles with dynamic margins for optimal recognition accuracy[cite: 8].
- **Smart Tracking Continuity:** Calculates Euclidean distance between consecutive frames so bounding boxes smoothly follow moving targets[cite: 11].

---

## 🛠️ Tech Stack & Architecture
- **Language:** Python 3
- **Libraries:** OpenCV, DeepFace, NumPy[cite: 8, 11]
- **AI Models Supported:** ArcFace, FaceNet, DeepFace[cite: 8, 11, 12]
- **Target Hardware:** USB Webcam / CCTV Camera, NVIDIA Jetson Nano / PC[cite: 8, 10]

---

## 📁 Repository Structure & File Overview
* **`capture_aadarsh.py` / `collect_faces.py`**: Handles student enrollment. Captures face samples from the webcam, automatically applies a 20% margin around the face box for DeepFace alignment, and saves them into the `dataset/` directory under the student's ID[cite: 8].
* **`recognize.py`**: The core execution script. Starts the camera stream, detects faces, handles asynchronous multi-threaded ArcFace recognition, and draws visual feedback (Green for Recognized, Red for Unknown)[cite: 11].
* **`train.py` / `debug_compare.py`**: Utility scripts to force-trigger DeepFace scans, compute similarity distances, and pre-build the vector representation database cache[cite: 9, 12, 13].

---

## 💻 Setup & Installation
### 1. Clone the repository
```bash
git clone [https://github.com/AKChauhan45/attendance_system_integration_with_ai__iot_embedded_system.git](https://github.com/AKChauhan45/attendance_system_integration_with_ai__iot_embedded_system.git)
cd attendance_system_integration_with_ai__iot_embedded_system
2. Install Dependencies
Bash
pip install opencv-python deepface tf-keras
🤖 How To Run
Step 1: Register/Enroll a Student
Bash
python collect_faces.py
Step 2: Pre-build the Database
Bash
python train.py
Step 3: Run Live Attendance System
Bash
python recognize.py
