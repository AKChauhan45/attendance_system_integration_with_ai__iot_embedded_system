# Smart Attendance System with AI & IoT Embedded System

An advanced, real-time automated attendance system utilizing **Computer Vision (OpenCV)** and **Deep Learning (DeepFace)**[cite: 1, 4]. The system features asynchronous multi-threaded face tracking and recognition, designed to run smoothly on embedded platforms like the **NVIDIA Jetson Nano**.

## 🚀 Key Features
- **Real-time Face Tracking:** Uses OpenCV Haar Cascades for fast, low-latency face detection[cite: 1, 4].
- **Asynchronous Deep Learning Inference:** Employs multi-threading to run heavy DeepFace facial recognition (ArcFace/FaceNet models) in the background, preventing camera feed lagging or flickering[cite: 4, 5].
- **Automated Dataset Collection:** Quick script to register new students by capturing multiple angles with dynamic margins for optimal recognition accuracy.
- **Smart Tracking Continuity:** Calculates Euclidean distance between consecutive frames so bounding boxes smoothly follow moving targets.

---

## 🛠️ Tech Stack & Architecture
- **Language:** Python 3
- **Libraries:** OpenCV, DeepFace, NumPy[cite: 1, 4]
- **AI Models Supported:** ArcFace, FaceNet, DeepFace[cite: 1, 4, 5]
- **Target Hardware:** USB Webcam / CCTV Camera, NVIDIA Jetson Nano / PC[cite: 1, 3]

---

## 📁 Repository Structure & File Overview

*   **`capture_aadhaar.py` / `collect_faces.py`**: Handles student enrollment. Captures face samples from the webcam, automatically applies a 20% margin around the face box for DeepFace alignment, and saves them into the `dataset/` directory under the student's ID[cite: 1].
*   **`recognize.py`**: The core execution script. Starts the camera stream, detects faces, handles asynchronous multi-threaded ArcFace recognition, and draws visual feedback (Green for Recognized, Red for Unknown).
*   **`train.py` / `debug_compare.py` / `rebuild_db.py`**: Utility scripts to force-trigger DeepFace scans, compute cosine/euclidean similarity distances, and pre-build the vector representation database cache (`.pkl` representations)[cite: 2, 5, 6].

---

## 💻 Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/AI-IoT-Attendance-System.git](https://github.com/YOUR_USERNAME/AI-IoT-Attendance-System.git)
cd AI-IoT-Attendance-System
