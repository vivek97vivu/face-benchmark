import os
import shutil

# Main dataset folder
SOURCE_DIR = "/home/vivek/Downloads/archive (1)/STAR-dataset/actors/presence7k"

# Destination folder
DEST_DIR = os.path.join(SOURCE_DIR, "AllImages")

# Image extensions to include
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tiff",
    ".tif",
}

# Create destination folder
os.makedirs(DEST_DIR, exist_ok=True)

count = 0

for root, dirs, files in os.walk(SOURCE_DIR):

    # Don't search inside AllImages
    if os.path.abspath(root) == os.path.abspath(DEST_DIR):
        continue

    for filename in files:
        extension = os.path.splitext(filename)[1].lower()

        if extension not in IMAGE_EXTENSIONS:
            continue

        source_file = os.path.join(root, filename)

        # Initial destination
        destination_file = os.path.join(DEST_DIR, filename)

        # Handle duplicate filenames
        name, ext = os.path.splitext(filename)
        duplicate_number = 1

        while os.path.exists(destination_file):
            new_filename = f"{name}_{duplicate_number}{ext}"
            destination_file = os.path.join(DEST_DIR, new_filename)
            duplicate_number += 1

        # Copy image
        shutil.copy2(source_file, destination_file)

        count += 1
        print(f"[{count}] {filename} -> {os.path.basename(destination_file)}")

print("\n================================")
print(f"Done! Copied {count} images.")
print(f"Images saved to: {DEST_DIR}")
print("================================")
