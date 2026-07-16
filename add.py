import cv2
import os

name = input("Enter student's ID/Name: ").strip()
save_path = os.path.join("dataset", name)
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"Adding faces for: {name}")
print("Press SPACE to capture | Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    display_frame = frame.copy()
    
    largest_face = None
    max_area = 0
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if w*h > max_area:
            max_area = w*h
            largest_face = (x, y, w, h)
            
    cv2.imshow("Collect Faces (Press SPACE to capture largest face)", display_frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        if largest_face is not None:
            x, y, w, h = largest_face
            # crop with 20% margin for DeepFace accuracy
            margin_x, margin_y = int(w * 0.2), int(h * 0.2)
            x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
            x2, y2 = min(frame.shape[1], x + w + margin_x), min(frame.shape[0], y + h + margin_y)
            
            face_crop = frame[y1:y2, x1:x2]
            
            img_path = os.path.join(save_path, f"{count}.jpg")
            cv2.imwrite(img_path, face_crop)
            print(f"Saved {img_path}")
            count += 1
        else:
            print("No face detected! Please adjust camera.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()