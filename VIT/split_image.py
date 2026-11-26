import os
import shutil

# CHANGE THESE PATHS
SOURCE_FOLDER = "/home/vivek/Music/face/All_Images"
OUTPUT_FOLDER = "/home/vivek/Music/face/Dataset"

IMAGES_PER_FOLDER = 5000

EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff", ".gif"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Collect images recursively
images = []

for root, dirs, files in os.walk(SOURCE_FOLDER):
    for file in files:
        if file.lower().endswith(EXTENSIONS):
            images.append(os.path.join(root, file))

images.sort()

total_images = len(images)

print(f"Total images found: {total_images:,}")
print(f"Expected folders: {(total_images + IMAGES_PER_FOLDER - 1) // IMAGES_PER_FOLDER}")

# Split and copy
for index, image_path in enumerate(images):

    folder_number = (index // IMAGES_PER_FOLDER) + 1

    folder_name = f"Batch_{folder_number:03d}"

    destination = os.path.join(OUTPUT_FOLDER, folder_name)

    os.makedirs(destination, exist_ok=True)

    filename = os.path.basename(image_path)

    shutil.copy2(
        image_path,
        os.path.join(destination, filename)
    )

    if (index + 1) % 1000 == 0:
        print(f"Processed {index + 1:,}/{total_images:,}")

print("\nDataset splitting completed!")
print(f"Total images: {total_images:,}")
print(f"Total folders: {(total_images + IMAGES_PER_FOLDER - 1) // IMAGES_PER_FOLDER}")