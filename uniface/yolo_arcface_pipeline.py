import os
import time
import cv2
import torch
import numpy as np
import pandas as pd
import onnxruntime as ort

from pathlib import Path
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("/home/vivek/Music/face/employee")

YOLO_MODEL = "/home/vivek/Downloads/yolo26s-face.pt"

ARCFACE_MODEL = (
    "/home/vivek/.insightface/models/buffalo_m/w600k_r50.onnx"
)

OUTPUT_DIR = Path(
    "/home/vivek/Music/face/yolo_arcface_results"
)

ALIGNED_DIR = OUTPUT_DIR / "aligned_faces"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
REPORT_DIR = OUTPUT_DIR / "reports"
BENCHMARK_DIR = OUTPUT_DIR / "benchmark"

# Set to 20 for initial testing.
# Set to None to process the complete employee folder.
MAX_IMAGES = None

IMAGE_SIZE = 640
DETECTION_CONFIDENCE = 0.70

GPU_ID = 0

# Only process images with exactly one detected face.
# Images with multiple faces are flagged for manual review.
REQUIRE_SINGLE_FACE = True

SAVE_ALIGNED_FACES = True

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [
    OUTPUT_DIR,
    ALIGNED_DIR,
    EMBEDDING_DIR,
    REPORT_DIR,
    BENCHMARK_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HARDWARE INFORMATION
# ============================================================

print("=" * 65)
print("YOLO26s-FACE + ARCFACE W600K PIPELINE")
print("=" * 65)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(GPU_ID))

print("ONNX Runtime:", ort.__version__)


# ============================================================
# LOAD YOLO FACE DETECTOR
# ============================================================

print("\nLoading YOLO face detector...")

detector = YOLO(YOLO_MODEL)

detector.to(f"cuda:{GPU_ID}")

print("YOLO task:", detector.task)
print("YOLO classes:", detector.names)


# ============================================================
# LOAD ARCFACE W600K
# ============================================================

print("\nLoading ArcFace W600K...")

providers = [
    ("CUDAExecutionProvider", {"device_id": GPU_ID}),
    "CPUExecutionProvider"
]

recognition_session = ort.InferenceSession(
    ARCFACE_MODEL,
    providers=providers
)

active_providers = recognition_session.get_providers()

print("ArcFace providers:", active_providers)

if "CUDAExecutionProvider" not in active_providers:
    raise RuntimeError(
        "ArcFace CUDA provider is not active. "
        "Check LD_LIBRARY_PATH before running."
    )

input_name = recognition_session.get_inputs()[0].name
output_name = recognition_session.get_outputs()[0].name

print("ArcFace input:", recognition_session.get_inputs()[0].shape)
print("ArcFace output:", recognition_session.get_outputs()[0].shape)


# ============================================================
# ARCFACE FIVE-POINT LANDMARK TEMPLATE
# ============================================================

# Standard InsightFace ArcFace 112x112 template.
# Order:
# 0. Left eye
# 1. Right eye
# 2. Nose
# 3. Left mouth corner
# 4. Right mouth corner

ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


# ============================================================
# FACE ALIGNMENT
# ============================================================

def align_face(image, landmarks):

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    if landmarks.shape != (5, 2):
        raise ValueError(
            f"Expected 5x2 landmarks, got {landmarks.shape}"
        )

    if not np.all(np.isfinite(landmarks)):
        raise ValueError("Invalid landmark coordinates")

    # Estimate similarity transform.
    matrix, inliers = cv2.estimateAffinePartial2D(
        landmarks,
        ARCFACE_TEMPLATE,
        method=cv2.LMEDS
    )

    if matrix is None or not np.all(np.isfinite(matrix)):
        raise RuntimeError("Face alignment failed")

    aligned = cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if aligned is None or aligned.size == 0:
        raise RuntimeError("Empty aligned face")

    return aligned


# ============================================================
# ARCFACE EMBEDDING GENERATION
# ============================================================

def generate_embedding(aligned_face):

    # InsightFace ArcFace convention:
    # BGR image, normalized to approximately [-1, 1].
    blob = cv2.dnn.blobFromImage(
        aligned_face,
        scalefactor=1.0 / 127.5,
        size=(112, 112),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
        crop=False
    )

    output = recognition_session.run(
        [output_name],
        {input_name: blob}
    )[0]

    embedding = output.reshape(-1).astype(np.float32)

    # L2 normalization.
    norm = np.linalg.norm(embedding)

    if norm < 1e-12:
        raise RuntimeError("Invalid zero embedding")

    embedding = embedding / norm

    if embedding.shape != (512,):
        raise RuntimeError(
            f"Expected 512 dimensions, got {embedding.shape}"
        )

    return embedding


# ============================================================
# LOAD EMPLOYEE IMAGES
# ============================================================

image_paths = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
])

if MAX_IMAGES is not None:
    image_paths = image_paths[:MAX_IMAGES]

print("\nEmployee image directory:", IMAGE_DIR)
print("Images to process:", len(image_paths))

if not image_paths:
    raise RuntimeError("No images found in employee directory")


# ============================================================
# WARMUP
# ============================================================

print("\nWarming up models...")

dummy = np.zeros((640, 640, 3), dtype=np.uint8)

for _ in range(5):
    detector.predict(
        dummy,
        imgsz=IMAGE_SIZE,
        conf=DETECTION_CONFIDENCE,
        device=GPU_ID,
        verbose=False
    )

dummy_face = np.zeros((112, 112, 3), dtype=np.uint8)

for _ in range(5):
    generate_embedding(dummy_face)

print("Warmup completed.")


# ============================================================
# BENCHMARK STORAGE
# ============================================================

embeddings = []
employee_names = []
embedding_metadata = []

image_results = []

total_detection_time = 0.0
total_alignment_time = 0.0
total_recognition_time = 0.0

total_faces_detected = 0
successful_images = 0
failed_images = 0
multiple_face_images = 0


# ============================================================
# PROCESS EMPLOYEE IMAGES
# ============================================================

print("\nStarting employee processing...\n")

overall_start = time.perf_counter()

for index, image_path in enumerate(image_paths, start=1):

    image_start = time.perf_counter()

    employee_id = image_path.stem

    detection_ms = 0.0
    alignment_ms = 0.0
    recognition_ms = 0.0

    status = "FAILED"
    detected_faces = 0
    confidence = None
    error_message = ""

    try:

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Could not read image")

        # ----------------------------------------------------
        # YOLO FACE DETECTION
        # ----------------------------------------------------

        detection_start = time.perf_counter()

        results = detector.predict(
            image,
            imgsz=IMAGE_SIZE,
            conf=DETECTION_CONFIDENCE,
            device=GPU_ID,
            verbose=False
        )

        detection_ms = (
            time.perf_counter() - detection_start
        ) * 1000

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:

            status = "NO_FACE_DETECTED"
            failed_images += 1

            image_results.append({
                "image": image_path.name,
                "employee_id": employee_id,
                "detected_faces": 0,
                "confidence": None,
                "detection_ms": detection_ms,
                "alignment_ms": 0,
                "recognition_ms": 0,
                "total_ms": (
                    time.perf_counter() - image_start
                ) * 1000,
                "status": status,
                "error": ""
            })

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name}: NO FACE"
            )

            continue

        detected_faces = len(result.boxes)
        total_faces_detected += detected_faces

        # ----------------------------------------------------
        # MULTIPLE FACE HANDLING
        # ----------------------------------------------------

        if REQUIRE_SINGLE_FACE and detected_faces != 1:

            status = "MULTIPLE_FACES"
            multiple_face_images += 1
            failed_images += 1

            image_results.append({
                "image": image_path.name,
                "employee_id": employee_id,
                "detected_faces": detected_faces,
                "confidence": None,
                "detection_ms": detection_ms,
                "alignment_ms": 0,
                "recognition_ms": 0,
                "total_ms": (
                    time.perf_counter() - image_start
                ) * 1000,
                "status": status,
                "error": "Manual review required"
            })

            print(
                f"[{index}/{len(image_paths)}] "
                f"{image_path.name}: "
                f"{detected_faces} FACES - REVIEW"
            )

            continue

        # ----------------------------------------------------
        # SELECT HIGHEST CONFIDENCE FACE
        # ----------------------------------------------------

        confidences = result.boxes.conf.cpu().numpy()

        best_index = int(np.argmax(confidences))

        confidence = float(confidences[best_index])

        bbox = result.boxes.xyxy[
            best_index
        ].cpu().numpy()

        if result.keypoints is None:
            raise RuntimeError("YOLO did not return landmarks")

        landmarks = result.keypoints.xy[
            best_index
        ].cpu().numpy()

        # ----------------------------------------------------
        # CHECK LANDMARKS
        # ----------------------------------------------------

        if landmarks.shape != (5, 2):
            raise RuntimeError(
                f"Unexpected landmark shape: {landmarks.shape}"
            )

        # ----------------------------------------------------
        # FACE ALIGNMENT
        # ----------------------------------------------------

        alignment_start = time.perf_counter()

        aligned_face = align_face(
            image,
            landmarks
        )

        alignment_ms = (
            time.perf_counter() - alignment_start
        ) * 1000

        # ----------------------------------------------------
        # SAVE ALIGNED FACE
        # ----------------------------------------------------

        if SAVE_ALIGNED_FACES:

            aligned_path = (
                ALIGNED_DIR /
                f"{employee_id}_aligned.jpg"
            )

            cv2.imwrite(
                str(aligned_path),
                aligned_face
            )

        # ----------------------------------------------------
        # ARCFACE RECOGNITION
        # ----------------------------------------------------

        recognition_start = time.perf_counter()

        embedding = generate_embedding(aligned_face)

        recognition_ms = (
            time.perf_counter() - recognition_start
        ) * 1000

        # ----------------------------------------------------
        # STORE EMBEDDING
        # ----------------------------------------------------

        embedding_index = len(embeddings)

        embeddings.append(embedding)
        employee_names.append(employee_id)

        embedding_metadata.append({
            "embedding_index": embedding_index,
            "employee_id": employee_id,
            "image": image_path.name,
            "face_confidence": confidence,
            "embedding_dimension": len(embedding),
            "bbox_x1": float(bbox[0]),
            "bbox_y1": float(bbox[1]),
            "bbox_x2": float(bbox[2]),
            "bbox_y2": float(bbox[3])
        })

        successful_images += 1
        status = "SUCCESS"

        total_detection_time += detection_ms
        total_alignment_time += alignment_ms
        total_recognition_time += recognition_ms

        total_ms = (
            time.perf_counter() - image_start
        ) * 1000

        image_results.append({
            "image": image_path.name,
            "employee_id": employee_id,
            "detected_faces": detected_faces,
            "confidence": confidence,
            "detection_ms": detection_ms,
            "alignment_ms": alignment_ms,
            "recognition_ms": recognition_ms,
            "total_ms": total_ms,
            "status": status,
            "error": ""
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} | "
            f"Confidence: {confidence:.3f} | "
            f"Detect: {detection_ms:.2f} ms | "
            f"Recognize: {recognition_ms:.2f} ms | "
            f"SUCCESS"
        )

    except Exception as e:

        failed_images += 1
        error_message = str(e)

        total_ms = (
            time.perf_counter() - image_start
        ) * 1000

        image_results.append({
            "image": image_path.name,
            "employee_id": employee_id,
            "detected_faces": detected_faces,
            "confidence": confidence,
            "detection_ms": detection_ms,
            "alignment_ms": alignment_ms,
            "recognition_ms": recognition_ms,
            "total_ms": total_ms,
            "status": "ERROR",
            "error": error_message
        })

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name}: ERROR - {error_message}"
        )


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

overall_time = time.perf_counter() - overall_start

if embeddings:

    embedding_array = np.asarray(
        embeddings,
        dtype=np.float32
    )

    names_array = np.asarray(
        employee_names,
        dtype=str
    )

    np.save(
        EMBEDDING_DIR / "employee_embeddings.npy",
        embedding_array
    )

    np.save(
        EMBEDDING_DIR / "employee_names.npy",
        names_array
    )

    pd.DataFrame(
        embedding_metadata
    ).to_csv(
        EMBEDDING_DIR / "embedding_metadata.csv",
        index=False
    )

else:

    embedding_array = np.empty((0, 512), dtype=np.float32)

    np.save(
        EMBEDDING_DIR / "employee_embeddings.npy",
        embedding_array
    )

    np.save(
        EMBEDDING_DIR / "employee_names.npy",
        np.array([], dtype=str)
    )


# ============================================================
# SAVE REPORTS
# ============================================================

results_df = pd.DataFrame(image_results)

results_df.to_csv(
    REPORT_DIR / "image_results.csv",
    index=False
)

failed_df = results_df[
    results_df["status"] != "SUCCESS"
]

failed_df.to_csv(
    REPORT_DIR / "failed_images.csv",
    index=False
)


# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

successful_df = results_df[
    results_df["status"] == "SUCCESS"
]

if len(successful_df) > 0:

    detection_values = successful_df["detection_ms"].to_numpy()
    recognition_values = successful_df["recognition_ms"].to_numpy()
    total_values = successful_df["total_ms"].to_numpy()

    summary = {
        "total_images": len(image_paths),
        "successful_images": successful_images,
        "failed_images": failed_images,
        "multiple_face_images": multiple_face_images,
        "total_detected_faces": total_faces_detected,
        "embedding_count": len(embeddings),
        "embedding_dimensions": 512,
        "detection_avg_ms": detection_values.mean(),
        "detection_p50_ms": np.percentile(detection_values, 50),
        "detection_p95_ms": np.percentile(detection_values, 95),
        "recognition_avg_ms": recognition_values.mean(),
        "recognition_p50_ms": np.percentile(recognition_values, 50),
        "recognition_p95_ms": np.percentile(recognition_values, 95),
        "total_avg_ms": total_values.mean(),
        "total_p50_ms": np.percentile(total_values, 50),
        "total_p95_ms": np.percentile(total_values, 95),
        "total_processing_time_sec": overall_time,
        "overall_images_per_sec": (
            len(image_paths) / overall_time
        ) if overall_time > 0 else 0
    }

else:

    summary = {
        "total_images": len(image_paths),
        "successful_images": 0,
        "failed_images": failed_images,
        "total_processing_time_sec": overall_time
    }


pd.DataFrame(
    [summary]
).to_csv(
    BENCHMARK_DIR / "performance_summary.csv",
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 65)
print("PIPELINE COMPLETED")
print("=" * 65)

for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key}: {value:.3f}")
    else:
        print(f"{key}: {value}")

print("\nEmbedding shape:", embedding_array.shape)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")
print("- embeddings/employee_embeddings.npy")
print("- embeddings/employee_names.npy")
print("- embeddings/embedding_metadata.csv")
print("- reports/image_results.csv")
print("- reports/failed_images.csv")
print("- benchmark/performance_summary.csv")

print("\nDone!")