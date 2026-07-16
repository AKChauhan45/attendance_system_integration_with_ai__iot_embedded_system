import cv2
from deepface import DeepFace

print("Testing DeepFace find with skip detector")

# We will just run it on one of the dataset images
try:
    results = DeepFace.find(
        img_path="dataset/IEC2023071/0.jpg",
        db_path="dataset",
        model_name="Facenet",
        detector_backend="skip",
        enforce_detection=False,
        silent=True
    )
    print("Result len:", len(results))
    if len(results) > 0:
        print("Empty?:", results[0].empty)
        if not results[0].empty:
            print("Distance:", results[0].iloc[0]['distance'])
except Exception as e:
    print("Error:", e)
