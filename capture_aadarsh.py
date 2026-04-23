import cv2
import os

name = "Aadarsh"
save_path = os.path.join("dataset", name)

os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print("Press S to save image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture - Aadarsh", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        img_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Saved {img_path}")
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
