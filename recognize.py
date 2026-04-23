import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.find(
            img_path=frame,
            db_path="dataset",
            enforce_detection=False
        )

        if len(result[0]) > 0:
            best_match = result[0].iloc[0]
            identity_path = best_match["identity"]
            distance = best_match["distance"]

            name = identity_path.split("\\")[-2]

            if distance < 0.4:
                label = f"{name} ({round(distance,2)})"
            else:
                label = "Unknown"

        else:
            label = "Unknown"

    except:
        label = "No Face"

    cv2.putText(frame, label, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
