import os
import cv2
import time
import numpy as np

from ai_edge_litert.interpreter import Interpreter
from uniface import FaceAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

EMPLOYEE_DIR = "/home/vivek/Music/face/employee"

MODEL_PATH = "/home/vivek/Music/face/mobile_net/mobilefacenet_int8.tflite"

# Change this if your Brio 300 is another device.
CAMERA_DEVICE = 4

# IMPORTANT:
# This is only an initial value.
# We will tune it after seeing your real scores.
THRESHOLD = 0.4

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# MOBILEFACENET TFLITE
# ============================================================

print("=" * 70)
print("Loading MobileFaceNet INT8")
print("=" * 70)

interpreter = Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input shape :", input_details["shape"])
print("Input dtype :", input_details["dtype"])
print("Input quant :", input_details["quantization"])

print("Output shape:", output_details["shape"])
print("Output dtype:", output_details["dtype"])
print("Output quant:", output_details["quantization"])

input_scale, input_zero = input_details["quantization"]
output_scale, output_zero = output_details["quantization"]


# ============================================================
# FACE DETECTOR / LANDMARKS
# ============================================================

print()
print("=" * 70)
print("Loading Face Detector")
print("=" * 70)

analyzer = FaceAnalyzer()

print(analyzer)


# ============================================================
# STANDARD 5-POINT FACE TEMPLATE
#
# MobileFaceNet/ArcFace-style alignment:
#
# left eye
# right eye
# nose
# left mouth
# right mouth
# ============================================================

REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ],
    dtype=np.float32
)


# ============================================================
# ALIGN FACE
# ============================================================

def align_face(image, landmarks):

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.shape != (5, 2):
        return None

    transform, _ = cv2.estimateAffinePartial2D(
        landmarks,
        REFERENCE_LANDMARKS,
        method=cv2.LMEDS
    )

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        image,
        transform,
        (112, 112),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return aligned


# ============================================================
# MOBILEFACENET EMBEDDING
# ============================================================

def get_mobileface_embedding(face_image):

    # OpenCV = BGR
    # Model expects RGB
    rgb = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    # Convert to float [0,1]
    image = rgb.astype(
        np.float32
    ) / 255.0

    # Quantize according to model
    image = np.round(
        image / input_scale
    ) + input_zero

    image = np.clip(
        image,
        -128,
        127
    ).astype(
        np.int8
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    interpreter.set_tensor(
        input_details["index"],
        image
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details["index"]
    )

    # Remove batch dimension
    embedding = output[0].astype(
        np.float32
    )

    # Dequantize
    embedding = (
        embedding - output_zero
    ) * output_scale

    # L2 normalization
    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        return None

    embedding = embedding / norm

    return embedding


# ============================================================
# GET LARGEST FACE
# ============================================================

def get_largest_face(faces):

    if not faces:
        return None

    return max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0]) *
        (f.bbox[3] - f.bbox[1])
    )


# ============================================================
# LOAD EMPLOYEE DATABASE
# ============================================================

print()
print("=" * 70)
print("Building Employee Database")
print("=" * 70)

employee_embeddings = []
employee_ids = []
employee_names = []

files = sorted(
    f for f in os.listdir(EMPLOYEE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
)

print(
    "Employee images found:",
    len(files)
)

failed = 0

for filename in files:

    path = os.path.join(
        EMPLOYEE_DIR,
        filename
    )

    image = cv2.imread(path)

    if image is None:

        print(
            "[ERROR] Cannot read:",
            filename
        )

        failed += 1
        continue

    faces = analyzer.analyze(
        image
    )

    face = get_largest_face(
        faces
    )

    if face is None:

        print(
            "[NO FACE]",
            filename
        )

        failed += 1
        continue

    if face.landmarks is None:

        print(
            "[NO LANDMARKS]",
            filename
        )

        failed += 1
        continue

    aligned = align_face(
        image,
        face.landmarks
    )

    if aligned is None:

        print(
            "[ALIGN FAILED]",
            filename
        )

        failed += 1
        continue

    embedding = get_mobileface_embedding(
        aligned
    )

    if embedding is None:

        print(
            "[EMBEDDING FAILED]",
            filename
        )

        failed += 1
        continue

    # --------------------------------------------------------
    # Extract ID and name
    # --------------------------------------------------------

    basename = os.path.splitext(
        filename
    )[0]

    if "_" in basename:

        employee_id, employee_name = basename.split(
            "_",
            1
        )

    else:

        employee_id = basename
        employee_name = basename

    employee_embeddings.append(
        embedding
    )

    employee_ids.append(
        employee_id
    )

    employee_names.append(
        employee_name
    )

    print(
        f"[OK] {employee_id:<8} "
        f"{employee_name}"
    )


if not employee_embeddings:

    print()
    print(
        "ERROR: No employee embeddings were created."
    )

    exit(1)


employee_embeddings = np.asarray(
    employee_embeddings,
    dtype=np.float32
)


print()
print("=" * 70)
print("DATABASE READY")
print("=" * 70)

print(
    "Valid employees :",
    len(employee_names)
)

print(
    "Failed images   :",
    failed
)

print(
    "Embedding shape :",
    employee_embeddings.shape
)

print(
    "Threshold       :",
    THRESHOLD
)


# ============================================================
# CAMERA
# ============================================================

print()
print("=" * 70)
print("Opening Camera")
print("=" * 70)

print(
    "Camera device:",
    CAMERA_DEVICE
)

cap = cv2.VideoCapture(
    CAMERA_DEVICE,
    cv2.CAP_V4L2
)

if not cap.isOpened():

    print()
    print(
        "ERROR: Could not open camera:"
    )

    print(
        CAMERA_DEVICE
    )

    print()
    print(
        "Check your Brio 300 with:"
    )

    print(
        "v4l2-ctl --list-devices"
    )

    exit(1)


# Optional camera resolution
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

print("Camera started.")
print("Press Q to quit.")


# ============================================================
# FPS
# ============================================================

fps_start = time.time()
fps_frames = 0
fps = 0.0


# ============================================================
# CAMERA LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "Failed to read camera frame."
        )

        break

    fps_frames += 1

    faces = analyzer.analyze(
        frame
    )

    for face in faces:

        if face.landmarks is None:
            continue

        # ----------------------------------------------------
        # Align camera face
        # ----------------------------------------------------

        aligned = align_face(
            frame,
            face.landmarks
        )

        if aligned is None:
            continue

        # ----------------------------------------------------
        # MobileFaceNet embedding
        # ----------------------------------------------------

        embedding = get_mobileface_embedding(
            aligned
        )

        if embedding is None:
            continue

        # ----------------------------------------------------
        # Cosine similarity
        #
        # Both vectors are normalized.
        # Therefore dot product = cosine similarity.
        # ----------------------------------------------------

        scores = np.dot(
            employee_embeddings,
            embedding
        )

        best_index = int(
            np.argmax(scores)
        )

        best_score = float(
            scores[best_index]
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

            color = (
                0,
                255,
                0
            )

        else:

            status = "UNKNOWN"

            display_name = "UNKNOWN"

            color = (
                0,
                0,
                255
            )

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = map(
            int,
            face.bbox
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # ----------------------------------------------------
        # Display name
        # ----------------------------------------------------

        cv2.putText(
            frame,
            display_name,
            (x1, max(30, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        # ----------------------------------------------------
        # Display score
        # ----------------------------------------------------

        score_text = (
            f"{status} "
            f"{best_score:.3f}"
        )

        cv2.putText(
            frame,
            score_text,
            (x1, max(55, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        print(
            f"{status:<7} "
            f"{best_id:<8} "
            f"{best_name:<30} "
            f"score={best_score:.4f}"
        )

    # ========================================================
    # FPS
    # ========================================================

    elapsed = time.time() - fps_start

    if elapsed >= 1.0:

        fps = fps_frames / elapsed

        fps_frames = 0
        fps_start = time.time()

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

    cv2.imshow(
        "MobileFaceNet INT8 - Face Recognition",
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

print()
print("Camera stopped.")
print("Done.")