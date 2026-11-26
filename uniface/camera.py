import os
import cv2
import time
import numpy as np
from uniface import FaceAnalyzer


# ============================================================
# CONFIG
# ============================================================

EMPLOYEE_DIR = "/home/vivek/Music/face/employee"

# IMPORTANT:
# Start with this value only for testing.
# We will tune it after collecting real scores.
THRESHOLD = 0.40

CAMERA_ID = 4

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# MODEL
# ============================================================

print("=" * 70)
print("Loading Face Recognition Model")
print("=" * 70)

analyzer = FaceAnalyzer()

print(analyzer)
print()


# ============================================================
# FUNCTIONS
# ============================================================

def normalize_embedding(embedding):
    """Normalize embedding for cosine similarity."""

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


def extract_face_embedding(image):
    """
    Detect face and return the largest face embedding.
    """

    faces = analyzer.analyze(image)

    if len(faces) == 0:
        return None, None

    # Select largest face
    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )

    if face.embedding is None:
        return None, face

    embedding = normalize_embedding(
        face.embedding
    )

    return embedding, face


# ============================================================
# LOAD EMPLOYEE DATABASE
# ============================================================

print("=" * 70)
print("Loading Employee Database")
print("=" * 70)

employee_embeddings = []
employee_ids = []
employee_names = []

files = sorted(
    f for f in os.listdir(EMPLOYEE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
)

print(f"Images found: {len(files)}")
print()

failed = 0

for filename in files:

    path = os.path.join(
        EMPLOYEE_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:
        print(f"[ERROR] Cannot read: {filename}")
        failed += 1
        continue

    embedding, face = extract_face_embedding(image)

    if embedding is None:
        print(f"[NO FACE] {filename}")
        failed += 1
        continue

    # --------------------------------------------------------
    # Extract employee information from filename
    # --------------------------------------------------------

    basename = os.path.splitext(filename)[0]

    if "_" in basename:

        employee_id, employee_name = basename.split(
            "_",
            1
        )

    else:

        employee_id = basename
        employee_name = basename

    employee_embeddings.append(embedding)
    employee_ids.append(employee_id)
    employee_names.append(employee_name)

    print(
        f"[OK] "
        f"{employee_id:<8} "
        f"{employee_name}"
    )


if len(employee_embeddings) == 0:

    print("\nERROR: No valid employee faces found.")

    exit(1)


# Convert to NumPy array
employee_embeddings = np.asarray(
    employee_embeddings,
    dtype=np.float32
)

print()
print("=" * 70)
print("Employee Database Ready")
print("=" * 70)

print(
    f"Valid employees : {len(employee_names)}"
)

print(
    f"Failed images   : {failed}"
)

print(
    f"Embedding shape : {employee_embeddings.shape}"
)

print()


# ============================================================
# CAMERA
# ============================================================

print("=" * 70)
print("Starting Camera")
print("=" * 70)

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print("ERROR: Cannot open camera.")

    exit(1)


print("Camera started.")
print("Press Q to quit.")
print()


# ============================================================
# FPS VARIABLES
# ============================================================

frame_count = 0
fps = 0

start_time = time.time()


# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to read camera frame.")
        break


    frame_count += 1

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------

    faces = analyzer.analyze(frame)


    for face in faces:

        if face.embedding is None:
            continue


        embedding = normalize_embedding(
            face.embedding
        )

        if embedding is None:
            continue


        # ----------------------------------------------------
        # Compare against all employees
        # ----------------------------------------------------

        similarities = np.dot(
            employee_embeddings,
            embedding
        )


        best_index = int(
            np.argmax(similarities)
        )

        best_score = float(
            similarities[best_index]
        )


        best_id = employee_ids[
            best_index
        ]

        best_name = employee_names[
            best_index
        ]


        # ----------------------------------------------------
        # Known / Unknown
        # ----------------------------------------------------

        if best_score >= THRESHOLD:

            status = "KNOWN"

            display_name = (
                f"{best_id} - {best_name}"
            )

        else:

            status = "UNKNOWN"

            display_name = "Unknown"


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )


        # ----------------------------------------------------
        # Draw
        # ----------------------------------------------------

        if status == "KNOWN":

            box_color = (
                0,
                255,
                0
            )

        else:

            box_color = (
                0,
                0,
                255
            )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        text1 = display_name

        text2 = (
            f"{status} "
            f"Similarity: {best_score:.3f}"
        )


        cv2.putText(
            frame,
            text1,
            (x1, max(25, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )


        cv2.putText(
            frame,
            text2,
            (x1, max(50, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )


        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        print(
            f"{status:<7} "
            f"{best_id:<8} "
            f"{best_name:<35} "
            f"score={best_score:.4f}"
        )


    # ========================================================
    # FPS
    # ========================================================

    elapsed = time.time() - start_time

    if elapsed >= 1.0:

        fps = frame_count / elapsed

        frame_count = 0

        start_time = time.time()


    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"Threshold: {THRESHOLD:.2f}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Face Recognition",
        frame
    )


    # Q = quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nCamera stopped.")
print("Done.")
