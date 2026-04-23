from deepface import DeepFace

print("Training started...")

DeepFace.find(
    img_path="dataset/Aditya/0.jpg",
    db_path="dataset",
    enforce_detection=False
)

print("Database created successfully!")

