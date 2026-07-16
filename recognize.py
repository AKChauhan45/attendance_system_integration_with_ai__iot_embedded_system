import cv2
import os
import threading
from deepface import DeepFace

print("Starting Camera... Press 'q' to exit.")
cap = cv2.VideoCapture(0)

# Load OpenCV face detector for lightning-fast tracking
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Format: dict mapping (cx, cy) -> {"name": str, "color": tuple, "ttl": int}
recognized_faces = [] 
is_recognizing = False

def recognize_worker(frame_copy):
    global recognized_faces, is_recognizing
    try:
        results = DeepFace.find(
            img_path=frame_copy, 
            db_path="dataset", 
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=False, 
            silent=True
        )
        
        new_recognized = []
        for df in results:
            if not df.empty:
                best = df.iloc[0]
                dist = best["distance"]
                
                # ArcFace's optimal cosine distance threshold is ~0.68.
                if dist <= 0.68:
                    identity = best["identity"].replace("\\", "/")
                    name = identity.split("/")[-2]
                    
                    x = int(best['source_x'])
                    y = int(best['source_y'])
                    w = int(best['source_w'])
                    h = int(best['source_h'])
                    
                    cx, cy = x + w//2, y + h//2
                    new_recognized.append({
                        "name": f"{name} ({round(dist,2)})", 
                        "color": (0, 255, 0), 
                        "center": (cx, cy)
                    })
        
        # Keep old recognized faces that aren't overwritten to prevent flickering
        recognized_faces = new_recognized
            
    except Exception as e:
        pass
    finally:
        is_recognizing = False

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Real-time Face Tracking
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # 2. Asynchronous Recognition
    if not is_recognizing:
        is_recognizing = True
        thread = threading.Thread(target=recognize_worker, args=(frame.copy(),))
        thread.daemon = True
        thread.start()

    # Sort faces by size (largest first) to prioritize closer people
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)

    # 3. Drawing and Label Matching
    # Add face count text
    cv2.putText(frame, f"Faces Detected: {len(faces)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    for (x, y, w, h) in faces:
        cx, cy = x + w//2, y + h//2
        label = "Unknown"
        color = (0, 0, 255) # Red for unknown
        
        best_match = None
        min_dist_sq = float('inf')

        # Find the closest matching recognized face
        for rec in recognized_faces:
            rx, ry = rec["center"]
            dist_sq = (cx - rx)**2 + (cy - ry)**2
            if dist_sq < (w//2)**2 and dist_sq < min_dist_sq: 
                best_match = rec
                min_dist_sq = dist_sq

        if best_match:
            label = best_match["name"]
            color = best_match["color"]
            # Update center so the label follows smoothly as they move
            best_match["center"] = (cx, cy)

        # Draw the box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        text_y = y - 10 if y > 20 else y + 20
        cv2.putText(frame, label, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()