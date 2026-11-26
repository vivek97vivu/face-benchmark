import os
import cv2
import math
from PIL import Image, ImageDraw

SOURCE = "/home/vivek/Music/face/failed_images/NO_FACE_DETECTED"

OUTPUT = "/home/vivek/Music/face/failed_contact_sheet.jpg"

files = sorted(os.listdir(SOURCE))

files = [
    f for f in files
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
]

thumb_w = 220
thumb_h = 250

cols = 4
rows = math.ceil(len(files) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * thumb_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

for i, filename in enumerate(files):

    path = os.path.join(SOURCE, filename)

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    img = Image.fromarray(image)

    img.thumbnail((thumb_w - 10, thumb_h - 45))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    sheet.paste(img, (x, y))

    draw.text(
        (x + 5, y + thumb_h - 40),
        filename[:30],
        fill="black"
    )

sheet.save(OUTPUT)

print("Contact sheet saved:", OUTPUT)