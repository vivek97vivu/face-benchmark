import os

# ============================================================
# FORCE CPU
# Raspberry Pi 5 deployment will use CPU
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import time
import pickle
import re
import numpy as np

from pathlib import Path
from deepface import DeepFace


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/home/vivek/Music/face")

EMPLOYEE_DIR = BASE_DIR / "employee"

CACHE_FILE = BASE_DIR / "deepface" / "ghostface_employee_embeddings.pkl"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "GhostFaceNet"

# OpenCV is lightweight and available in your current setup.
DETECTOR_BACKEND = "opencv"


# ============================================================
# RECOGNITION THRESHOLD
# ============================================================

# This is only an INITIAL threshold.
# We will tune it after collecting real scores.
THRESHOLD = 0.2


# ============================================================
# CAMERA
# ============================================================

# Change this if your Brio 300 is on another device.
#
# Examples:
# /dev/video0
# /dev/video1
# /dev/video2
#
CAMERA_DEVICE = 2

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# OPENCV FACE DETECTOR
# ============================================================

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)


# ============================================================
# EMPLOYEE NAME PARSER
# ============================================================

def parse_employee(filename):

    stem = Path(filename).stem

    # Example:
    #
    # 10001_Prasanth Kuniyil.jpg
    #
    # ID   = 10001
    # Name = Prasanth Kuniyil

    match = re.match(r"^(\d+)_(.+)$", stem)

    if match:

        employee_id = match.group(1)
        employee_name = match.group(2)

    else:

        # Example:
        #
        # vivek.jpg
        #
        employee_id = "N/A"
        employee_name = stem

    return employee_id, employee_name


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(embedding)

    if norm == 0:

        return embedding

    return embedding / norm


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = normalize_embedding(a)
    b = normalize_embedding(b)

    return float(np.dot(a, b))


# ============================================================
# GENERATE GHOSTFACENET EMBEDDING
# ============================================================

def get_embedding(image):

    result = DeepFace.represent(
        img_path=str(image),
        model_name=MODEL_NAME,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=True,
        align=True,
    )

    if not result:

        raise RuntimeError("No face detected")

    embedding = result[0]["embedding"]

    return normalize_embedding(embedding)


# ============================================================
# BUILD EMPLOYEE DATABASE
# ============================================================

def build_database():

    print()
    print("=" * 70)
    print("BUILDING GHOSTFACENET EMPLOYEE DATABASE")
    print("=" * 70)

    files = sorted(
        [
            p
            for p in EMPLOYEE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    print("Employee images:", len(files))
    print()

    database = []

    total_start = time.perf_counter()

    for index, image_path in enumerate(files, start=1):

        employee_id, employee_name = parse_employee(
            image_path.name
        )

        print(
            f"[{index:3d}/{len(files)}] "
            f"{employee_id} - {employee_name}"
        )

        start = time.perf_counter()

        try:

            embedding = get_embedding(image_path)

            elapsed = time.perf_counter() - start

            database.append(
                {
                    "id": employee_id,
                    "name": employee_name,
                    "image": str(image_path),
                    "embedding": embedding,
                }
            )

            print(
                f"       OK "
                f"| {elapsed:.2f}s "
                f"| embedding={len(embedding)}"
            )

        except Exception as e:

            print(
                f"       FAILED: {e}"
            )

    total_time = time.perf_counter() - total_start

    print()
    print("=" * 70)
    print("DATABASE COMPLETE")
    print("=" * 70)

    print("Successful :", len(database))
    print("Failed     :", len(files) - len(database))
    print("Total time :", f"{total_time:.2f}s")

    return database


# ============================================================
# SAVE DATABASE
# ============================================================

def save_database(database):

    with open(CACHE_FILE, "wb") as f:

        pickle.dump(
            database,
            f
        )

    print()
    print("Database saved:")
    print(CACHE_FILE)


# ============================================================
# LOAD DATABASE
# ============================================================

def load_database():

    with open(CACHE_FILE, "rb") as f:

        database = pickle.load(f)

    return database


# ============================================================
# FIND BEST EMPLOYEE
# ============================================================

def recognize_face(face_image, database):

    try:

        start = time.perf_counter()

        # Because face_image is already cropped,
        # tell DeepFace to skip another detector.
        result = DeepFace.represent(
            img_path=face_image,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )

        if not result:

            return None

        query_embedding = normalize_embedding(
            result[0]["embedding"]
        )

        embedding_time = time.perf_counter() - start

    except Exception:

        return None

    best_score = -1.0
    best_employee = None

    for employee in database:

        score = cosine_similarity(
            query_embedding,
            employee["embedding"]
        )

        if score > best_score:

            best_score = score
            best_employee = employee

    return (
        best_employee,
        best_score,
        embedding_time,
    )


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    print()
    print("=" * 70)
    print("OPENING CAMERA")
    print("=" * 70)

    print("Camera:", CAMERA_DEVICE)

    camera = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    camera.set(
        cv2.CAP_PROP_FPS,
        30
    )

    if not camera.isOpened():

        print()
        print("ERROR: Cannot open camera.")
        print()
        print("Run:")
        print("    v4l2-ctl --list-devices")
        print()
        print("Then change CAMERA_DEVICE in this script.")

        return None

    print("Camera opened successfully.")

    return camera


# ============================================================
# LIVE RECOGNITION
# ============================================================

def run_camera(database):

    camera = open_camera()

    if camera is None:

        return

    print()
    print("=" * 70)
    print("GHOSTFACENET LIVE RECOGNITION")
    print("=" * 70)

    print()
    print("Press Q to quit.")
    print()

    frame_count = 0

    last_result = None

    last_embedding_time = 0

    # Process one face every N frames.
    # GhostFaceNet on CPU is not fast enough to process
    # every camera frame.
    PROCESS_EVERY_N_FRAMES = 5

    fps_start = time.perf_counter()
    fps_counter = 0
    fps = 0.0

    while True:

        ret, frame = camera.read()

        if not ret:

            print("Camera frame read failed.")

            break

        frame_count += 1
        fps_counter += 1

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = time.perf_counter() - fps_start

        if elapsed >= 1.0:

            fps = fps_counter / elapsed

            fps_counter = 0
            fps_start = time.perf_counter()

        # ----------------------------------------------------
        # Convert to grayscale for face detector
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # ----------------------------------------------------
        # Process faces periodically
        # ----------------------------------------------------

        if (
            len(faces) > 0
            and frame_count % PROCESS_EVERY_N_FRAMES == 0
        ):

            # Select largest face
            x, y, w, h = max(
                faces,
                key=lambda r: r[2] * r[3]
            )

            # Add margin around face
            margin = int(
                0.25 * max(w, h)
            )

            x1 = max(
                0,
                x - margin
            )

            y1 = max(
                0,
                y - margin
            )

            x2 = min(
                frame.shape[1],
                x + w + margin
            )

            y2 = min(
                frame.shape[0],
                y + h + margin
            )

            face_crop = frame[
                y1:y2,
                x1:x2
            ]

            if face_crop.size > 0:

                result = recognize_face(
                    face_crop,
                    database
                )

                if result is not None:

                    (
                        employee,
                        score,
                        embedding_time,
                    ) = result

                    last_result = (
                        employee,
                        score
                    )

                    last_embedding_time = (
                        embedding_time
                    )

        # ----------------------------------------------------
        # Draw results
        # ----------------------------------------------------

        for x, y, w, h in faces:

            if last_result is None:

                label = "FACE"

            else:

                employee, score = last_result

                if score >= THRESHOLD:

                    label = (
                        f"KNOWN: "
                        f"{employee['name']} "
                        f"{score:.3f}"
                    )

                else:

                    label = (
                        f"UNKNOWN "
                        f"{score:.3f}"
                    )

            # Bounding box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Background for text

            cv2.rectangle(
                frame,
                (x, max(0, y - 35)),
                (
                    x + 400,
                    y
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                label,
                (x + 5, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ----------------------------------------------------
        # Status information
        # ----------------------------------------------------

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Threshold: {THRESHOLD:.2f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"GhostFaceNet: 512-D",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            "GhostFaceNet Employee Recognition",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    camera.release()

    cv2.destroyAllWindows()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("GHOSTFACENET EMPLOYEE RECOGNITION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check employee folder
    # --------------------------------------------------------

    if not EMPLOYEE_DIR.exists():

        raise FileNotFoundError(
            f"Employee directory not found:\n{EMPLOYEE_DIR}"
        )

    # --------------------------------------------------------
    # Build database if cache doesn't exist
    # --------------------------------------------------------

    if CACHE_FILE.exists():

        print()
        print("Existing employee database found.")

        answer = input(
            "Rebuild employee database? [y/N]: "
        ).strip().lower()

        if answer == "y":

            database = build_database()

            save_database(database)

        else:

            print("Loading existing database...")

            database = load_database()

    else:

        print()
        print("No employee database found.")

        database = build_database()

        save_database(database)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EMPLOYEE DATABASE READY")
    print("=" * 70)

    print(
        "Employees loaded:",
        len(database)
    )

    # --------------------------------------------------------
    # Start camera
    # --------------------------------------------------------

    run_camera(database)
