import os
import shutil

source_folder = "/home/vivek/Downloads/archive(2)"
destination_folder = "/home/vivek/Music/face/All_Images"

extensions = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".gif"
)

os.makedirs(destination_folder, exist_ok=True)

count = 0

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith(extensions):
            source_path = os.path.join(root, file)

            name, ext = os.path.splitext(file)
            destination_path = os.path.join(destination_folder, file)

            index = 1
            while os.path.exists(destination_path):
                destination_path = os.path.join(
                    destination_folder,
                    f"{name}_{index}{ext}"
                )
                index += 1

            shutil.copy2(source_path, destination_path)
            count += 1

print(f"Total images copied: {count}")
print(f"Destination: {destination_folder}")