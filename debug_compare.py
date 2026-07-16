from deepface import DeepFace

result = DeepFace.find(
    img_path="dataset/Aditya/0.jpg",
    db_path="dataset",
    enforce_detection=False
)

print(result[0][["identity", "distance"]])

