import cv2
import os

name = "Aditya"   # change name for each student
save_path = f"dataset/{name}"

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press SPACE to capture image")
print("Press ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collect Faces", frame)
    key = cv2.waitKey(1)

    if key == 32:  # SPACE
        img_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1

    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
