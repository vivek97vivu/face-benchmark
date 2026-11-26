import cv2
import time
import gc
import numpy as np
import pandas as pd
import onnxruntime as ort

from pathlib import Path
from insightface.model_zoo import get_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ROOT = Path("/home/vivek/.insightface/models")

IMAGE_DIR = Path("/home/vivek/Music/face/employee")

OUTPUT_DIR = Path(
    "/home/vivek/Music/face/insightface_benchmark_results"
)

IMAGE_SIZE = (640, 640)

CONFIDENCE = 0.50
NMS_THRESHOLD = 0.40

WARMUP_RUNS = 10

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

# CUDA + CPU fallback.
# TensorRT is intentionally excluded for consistent CUDA testing.
AVAILABLE_PROVIDERS = ort.get_available_providers()

if "CUDAExecutionProvider" not in AVAILABLE_PROVIDERS:
    raise RuntimeError(
        "CUDAExecutionProvider is unavailable. "
        "Check your ONNX Runtime CUDA configuration."
    )

PROVIDERS = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

CTX_ID = 0


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "antelopev2": {
        "path": MODEL_ROOT / "antelopev2" / "scrfd_10g_bnkps.onnx",
        "type": "SCRFD 10G"
    },

    "buffalo_l": {
        "path": MODEL_ROOT / "buffalo_l" / "det_10g.onnx",
        "type": "SCRFD 10G"
    },

    "buffalo_m": {
        "path": MODEL_ROOT / "buffalo_m" / "det_2.5g.onnx",
        "type": "SCRFD 2.5G"
    },

    "buffalo_s": {
        "path": MODEL_ROOT / "buffalo_s" / "det_500m.onnx",
        "type": "SCRFD 500M"
    }
}


# ============================================================
# IMAGE LOADING
# ============================================================

def get_images():

    images = sorted([
        p for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    return images


# ============================================================
# BENCHMARK ONE MODEL
# ============================================================

def benchmark_model(model_name, model_info, image_paths):

    model_path = model_info["path"]

    print("\n" + "=" * 65)
    print(f"MODEL: {model_name}")
    print(f"TYPE: {model_info['type']}")
    print(f"PATH: {model_path}")
    print("=" * 65)

    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return None, None, None

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    detector = get_model(
        str(model_path),
        providers=PROVIDERS
    )

    detector.prepare(
        ctx_id=CTX_ID,
        det_thresh=CONFIDENCE,
        nms_thresh=NMS_THRESHOLD,
        input_size=IMAGE_SIZE
    )

    actual_providers = detector.session.get_providers()

    print("Available providers:", AVAILABLE_PROVIDERS)
    print("Actual session providers:", actual_providers)

    if "CUDAExecutionProvider" not in actual_providers:
        raise RuntimeError(
            f"{model_name} did not initialize CUDA."
        )

    print("CUDA initialization successful.")

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(f"\nRunning {WARMUP_RUNS} warm-up iterations...")

    warmup_image = None

    for image_path in image_paths:

        warmup_image = cv2.imread(str(image_path))

        if warmup_image is not None:
            break

    if warmup_image is None:
        raise RuntimeError("No readable images found.")

    for _ in range(WARMUP_RUNS):

        detector.detect(
            warmup_image,
            input_size=IMAGE_SIZE,
            max_num=0
        )

    print("Warm-up completed.")

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    results = []

    total_faces = 0
    detected_images = 0
    missed_images = 0
    unreadable_images = 0

    latencies = []

    total_images = len(image_paths)

    print(f"\nBenchmarking {total_images} images...\n")

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(str(image_path))

        if image is None:

            unreadable_images += 1

            results.append({
                "model": model_name,
                "image": image_path.name,
                "detections": 0,
                "max_confidence": 0,
                "largest_face_area": 0,
                "latency_ms": 0,
                "status": "UNREADABLE"
            })

            continue

        # Inference timing
        start = time.perf_counter()

        bboxes, keypoints = detector.detect(
            image,
            input_size=IMAGE_SIZE,
            max_num=0
        )

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

        # Detection information
        if bboxes is not None and len(bboxes) > 0:

            detection_count = len(bboxes)

            detected_images += 1
            total_faces += detection_count

            confidences = bboxes[:, 4]

            max_confidence = float(np.max(confidences))

            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]

            areas = widths * heights

            largest_face_area = float(np.max(areas))

            status = "DETECTED"

        else:

            detection_count = 0
            max_confidence = 0
            largest_face_area = 0

            missed_images += 1

            status = "NO_FACE_DETECTED"

        results.append({
            "model": model_name,
            "image": image_path.name,
            "detections": detection_count,
            "max_confidence": max_confidence,
            "largest_face_area": largest_face_area,
            "latency_ms": latency_ms,
            "status": status
        })

        if index % 100 == 0 or index == total_images:

            print(
                f"Processed {index}/{total_images} | "
                f"Detected: {detected_images} | "
                f"Missed: {missed_images}"
            )

    # --------------------------------------------------------
    # Calculate statistics
    # --------------------------------------------------------

    if not latencies:
        raise RuntimeError("No images were successfully benchmarked.")

    latencies = np.array(latencies)

    avg_latency = float(np.mean(latencies))
    p50_latency = float(np.percentile(latencies, 50))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))

    fps = 1000.0 / avg_latency

    valid_images = total_images - unreadable_images

    detection_coverage = (
        detected_images / valid_images * 100
        if valid_images else 0
    )

    summary = {
        "model": model_name,
        "detector_type": model_info["type"],
        "actual_providers": ",".join(actual_providers),
        "total_images": total_images,
        "valid_images": valid_images,
        "detected_images": detected_images,
        "missed_images": missed_images,
        "unreadable_images": unreadable_images,
        "detection_coverage_percent": detection_coverage,
        "total_faces": total_faces,
        "avg_latency_ms": avg_latency,
        "p50_latency_ms": p50_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "fps": fps
    }

    print("\nMODEL RESULTS")
    print("-" * 50)

    print(f"Images:             {total_images}")
    print(f"Detected images:    {detected_images}")
    print(f"Missed images:      {missed_images}")
    print(f"Unreadable images:  {unreadable_images}")
    print(f"Total faces:        {total_faces}")

    print(f"Coverage:           {detection_coverage:.2f}%")

    print(f"Average latency:    {avg_latency:.3f} ms")
    print(f"P50 latency:        {p50_latency:.3f} ms")
    print(f"P95 latency:        {p95_latency:.3f} ms")
    print(f"P99 latency:        {p99_latency:.3f} ms")

    print(f"FPS:                {fps:.2f}")

    print(f"Actual providers:   {actual_providers}")

    per_image_df = pd.DataFrame(results)

    del detector
    gc.collect()

    return summary, per_image_df, results


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    image_paths = get_images()

    print("=" * 65)
    print("INSIGHTFACE SCRFD DETECTOR BENCHMARK")
    print("=" * 65)

    print("ONNX Runtime:", ort.__version__)
    print("CUDA providers:", AVAILABLE_PROVIDERS)

    print("Image directory:", IMAGE_DIR)
    print("Total images:", len(image_paths))

    print("Image size:", IMAGE_SIZE)
    print("Confidence:", CONFIDENCE)
    print("NMS:", NMS_THRESHOLD)
    print("Warm-up:", WARMUP_RUNS)

    if not image_paths:
        raise RuntimeError("No images found.")

    summaries = []
    all_per_image = []

    # --------------------------------------------------------
    # Run each model
    # --------------------------------------------------------

    for model_name, model_info in MODELS.items():

        summary, per_image_df, results = benchmark_model(
            model_name,
            model_info,
            image_paths
        )

        if summary is not None:

            summaries.append(summary)
            all_per_image.append(per_image_df)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    if not summaries:
        raise RuntimeError("No models completed successfully.")

    summary_df = pd.DataFrame(summaries)

    per_image_df = pd.concat(
        all_per_image,
        ignore_index=True
    )

    missed_df = per_image_df[
        per_image_df["status"] == "NO_FACE_DETECTED"
    ]

    summary_path = OUTPUT_DIR / "insightface_summary.csv"

    per_image_path = OUTPUT_DIR / "insightface_per_image.csv"

    missed_path = OUTPUT_DIR / "insightface_missed_images.csv"

    summary_df.to_csv(summary_path, index=False)

    per_image_df.to_csv(per_image_path, index=False)

    missed_df.to_csv(missed_path, index=False)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON")
    print("=" * 90)

    columns = [
        "model",
        "detected_images",
        "missed_images",
        "detection_coverage_percent",
        "avg_latency_ms",
        "p95_latency_ms",
        "fps"
    ]

    print(
        summary_df[columns].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print("\nRESULT FILES")
    print("-" * 50)

    print(summary_path)
    print(per_image_path)
    print(missed_path)

    print("\nBENCHMARK COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    main()
