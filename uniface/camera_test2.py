import os
import cv2
import time
import shutil
import numpy as np
from collections import defaultdict
from uniface import FaceAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

EMPLOYEE_DIR = "/home/vivek/Music/face/employee"

# Failed images will be copied here
REPORT_DIR = "/home/vivek/Music/face/failed_images"

THRESHOLD = 0.1

CAMERA_ID = 4

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff"
)


# ============================================================
# INITIALIZATION
# ============================================================

print("=" * 70)
print("UNIFACE EMPLOYEE FACE RECOGNITION")
print("=" * 70)

print("\nLoading Face Recognition Model...")

analyzer = FaceAnalyzer()

print(analyzer)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# FAILURE CATEGORIES
# ============================================================

failure_categories = [
    "UNREADABLE",
    "NO_FACE_DETECTED",
    "EMBEDDING_MISSING",
    "INVALID_EMBEDDING",
    "ZERO_NORM",
    "EXCEPTION"
]

failed_images = defaultdict(list)


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    if not np.all(np.isfinite(embedding)):
        return None

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None

    return embedding / norm


# ============================================================
# EXTRACT FACE EMBEDDING
# ============================================================

def extract_face_embedding(image):

    try:

        faces = analyzer.analyze(image)

        if faces is None or len(faces) == 0:

            return None, None, "NO_FACE_DETECTED"

        # Select largest face

        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        if face.embedding is None:

            return None, face, "EMBEDDING_MISSING"

        embedding = np.asarray(
            face.embedding,
            dtype=np.float32
        )

        if not np.all(np.isfinite(embedding)):

            return None, face, "INVALID_EMBEDDING"

        norm = np.linalg.norm(embedding)

        if norm == 0:

            return None, face, "ZERO_NORM"

        embedding = embedding / norm

        return embedding, face, "SUCCESS"

    except Exception as e:

        return None, None, f"EXCEPTION: {str(e)}"


# ============================================================
# LOAD EMPLOYEE DATABASE
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMPLOYEE DATABASE")
print("=" * 70)

employee_embeddings = []
employee_ids = []
employee_names = []

files = sorted(
    f for f in os.listdir(EMPLOYEE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
)

total_images = len(files)

print(f"\nTotal images found: {total_images}")

print("\nProcessing employee images...\n")


for index, filename in enumerate(files, 1):

    path = os.path.join(
        EMPLOYEE_DIR,
        filename
    )

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(path)

    if image is None:

        print(f"[UNREADABLE] {filename}")

        failed_images["UNREADABLE"].append(
            filename
        )

        continue

    # --------------------------------------------------------
    # EXTRACT EMBEDDING
    # --------------------------------------------------------

    embedding, face, reason = extract_face_embedding(
        image
    )

    if embedding is None:

        if reason.startswith("EXCEPTION:"):

            category = "EXCEPTION"

        else:

            category = reason

        print(f"[{reason}] {filename}")

        failed_images[category].append(
            filename
        )

        continue

    # --------------------------------------------------------
    # EXTRACT EMPLOYEE INFORMATION
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
        f"{employee_id:<10} "
        f"{employee_name}"
    )

    # Progress

    if index % 500 == 0:

        print(
            f"\nProgress: {index}/{total_images}\n"
        )


# ============================================================
# CONVERT DATABASE TO NUMPY
# ============================================================

if len(employee_embeddings) == 0:

    print("\nERROR: No valid employee embeddings found.")

    exit(1)


employee_embeddings = np.asarray(
    employee_embeddings,
    dtype=np.float32
)


# ============================================================
# FAILED IMAGE REPORT
# ============================================================

print("\n" + "=" * 70)
print("FAILED IMAGE ANALYSIS")
print("=" * 70)

total_failed = 0

for category in failure_categories:

    filenames = failed_images[category]

    count = len(filenames)

    total_failed += count

    print(
        f"{category:<25}: {count}"
    )


# ============================================================
# COPY FAILED IMAGES INTO SEPARATE FOLDERS
# ============================================================

print("\nOrganizing failed images...")

for category in failure_categories:

    filenames = failed_images[category]

    if not filenames:
        continue

    category_dir = os.path.join(
        REPORT_DIR,
        category
    )

    os.makedirs(
        category_dir,
        exist_ok=True
    )

    for filename in filenames:

        source = os.path.join(
            EMPLOYEE_DIR,
            filename
        )

        destination = os.path.join(
            category_dir,
            filename
        )

        try:

            shutil.copy2(
                source,
                destination
            )

        except Exception as e:

            print(
                f"[COPY ERROR] {filename}: {e}"
            )


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "failed_images_report.txt"
)

with open(report_path, "w") as f:

    f.write("UNIFACE FAILED IMAGE REPORT\n")

    f.write("=" * 60 + "\n")

    f.write(
        f"Total images: {total_images}\n"
    )

    f.write(
        f"Successful: {len(employee_embeddings)}\n"
    )

    f.write(
        f"Failed: {total_failed}\n\n"
    )

    for category in failure_categories:

        filenames = failed_images[category]

        f.write(
            f"\n{category} ({len(filenames)})\n"
        )

        f.write("-" * 50 + "\n")

        for filename in filenames:

            f.write(filename + "\n")


# ============================================================
# DATABASE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("EMPLOYEE DATABASE READY")
print("=" * 70)

print(
    f"Total images     : {total_images}"
)

print(
    f"Valid embeddings : {len(employee_embeddings)}"
)

print(
    f"Failed images    : {total_failed}"
)

print(
    f"Embedding shape  : {employee_embeddings.shape}"
)

print(
    f"Failure report   : {report_path}"
)

print(
    f"Failed image dir : {REPORT_DIR}"
)

print()


# ============================================================
# CAMERA INITIALIZATION
# ============================================================

print("=" * 70)
print("STARTING CAMERA")
print("=" * 70)

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    print("ERROR: Cannot open camera.")

    exit(1)


print("Camera started successfully.")

print("Press Q to quit.\n")


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
    # DETECT FACES
    # --------------------------------------------------------

    try:

        faces = analyzer.analyze(frame)

    except Exception as e:

        print(f"Camera detection error: {e}")

        faces = []

    # --------------------------------------------------------
    # PROCESS EACH FACE
    # --------------------------------------------------------

    for face in faces:

        if face.embedding is None:
            continue

        embedding = normalize_embedding(
            face.embedding
        )

        if embedding is None:
            continue

        # ----------------------------------------------------
        # COSINE SIMILARITY
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

        best_id = employee_ids[best_index]

        best_name = employee_names[best_index]

        # ----------------------------------------------------
        # KNOWN / UNKNOWN
        # ----------------------------------------------------

        if best_score >= THRESHOLD:

            status = "KNOWN"

            display_name = (
                f"{best_id} - {best_name}"
            )

            box_color = (0, 255, 0)

        else:

            status = "UNKNOWN"

            display_name = "Unknown"

            box_color = (0, 0, 255)

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        # ----------------------------------------------------
        # DISPLAY TEXT
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
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            f"{status:<7} "
            f"{best_id:<10} "
            f"{best_name:<35} "
            f"score={best_score:.4f}"
        )

    # ========================================================
    # FPS CALCULATION
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
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "UniFace Employee Recognition",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\n" + "=" * 70)
print("CAMERA STOPPED")
print("FACE RECOGNITION COMPLETED")
print("=" * 70)
