import os
from deepface import DeepFace

print("Rebuilding database...")

# This automatically finds the first image in your dataset folder
dataset_path = "dataset"
all_images = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            all_images.append(os.path.join(root, file))

if all_images:
    # Use the first image found to trigger the scan
    DeepFace.find(
        img_path=all_images[0],
        db_path=dataset_path,
        model_name="ArcFace",
        detector_backend="opencv",
        enforce_detection=False
    )
    print(f"Database created successfully using {all_images[0]} as a trigger!")
else:
    print("No images found in the dataset folder!")