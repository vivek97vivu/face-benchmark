import os
import time
import argparse
import traceback
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from uniface import FaceAnalyzer
from uniface.detection import SCRFD, RetinaFace
from uniface.constants import SCRFDWeights, RetinaFaceWeights


# ============================================================
# CONFIGURATION
# ============================================================

EMPLOYEE_DIR = Path("/home/vivek/Music/face/employee")
OUTPUT_DIR = Path("/home/vivek/Music/face/benchmark_results_demo")

IMAGE_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.50
NMS_THRESHOLD = 0.40

WARMUP_RUNS = 10

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

# Use the same execution provider preference for both detectors.
# CUDAExecutionProvider will be used when available.
PROVIDERS = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_images(folder):
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(images)


# ============================================================
# DETECTOR INITIALIZATION
# ============================================================

def initialize_detectors():

    print("\nInitializing detectors...")

    detectors = {}

    # --------------------------------------------------------
    # 1. Existing UniFace baseline
    # --------------------------------------------------------

    print("[1/3] Loading current UniFace FaceAnalyzer...")

    baseline = FaceAnalyzer()

    # Prefer the internal detector directly if this UniFace
    # version exposes it. This avoids including embedding
    # extraction in the baseline timing.
    internal_detector = getattr(baseline, "detector", None)

    if internal_detector is None:
        internal_detector = getattr(baseline, "_detector", None)

    if internal_detector is not None and hasattr(
        internal_detector, "detect"
    ):
        detectors["UniFace_Baseline"] = {
            "object": internal_detector,
            "mode": "detect",
        }
        print("    Using FaceAnalyzer's internal detector.")
    else:
        # Fallback: this measures the complete analyze() pipeline,
        # which may include alignment and embedding extraction.
        detectors["UniFace_Baseline"] = {
            "object": baseline,
            "mode": "analyze",
        }
        print(
            "    Internal detector not exposed; "
            "using FaceAnalyzer.analyze() fallback."
        )

    # --------------------------------------------------------
    # 2. SCRFD
    # --------------------------------------------------------

    print("[2/3] Loading SCRFD 500M...")

    scrfd = SCRFD(
        model_name=SCRFDWeights.SCRFD_500M_KPS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=IMAGE_SIZE,
        providers=PROVIDERS,
    )

    detectors["SCRFD_500M"] = {
        "object": scrfd,
        "mode": "detect",
    }

    # --------------------------------------------------------
    # 3. RetinaFace
    # --------------------------------------------------------

    print("[3/3] Loading RetinaFace MNET_V2...")

    retinaface = RetinaFace(
        model_name=RetinaFaceWeights.MNET_V2,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        input_size=IMAGE_SIZE,
        providers=PROVIDERS,
    )

    detectors["RetinaFace_MNET_V2"] = {
        "object": retinaface,
        "mode": "detect",
    }

    print("\nAll detectors initialized.")

    return detectors


# ============================================================
# DETECTION
# ============================================================

def run_detection(detector_info, image):

    detector = detector_info["object"]
    mode = detector_info["mode"]

    if mode == "detect":
        faces = detector.detect(image)

    elif mode == "analyze":
        faces = detector.analyze(image)

    else:
        raise ValueError(f"Unknown detector mode: {mode}")

    return faces


# ============================================================
# WARM-UP
# ============================================================

def warmup_detectors(detectors, sample_image):

    print("\nWarming up detectors...")

    for name, detector_info in detectors.items():

        print(f"Warm-up: {name}")

        for _ in range(WARMUP_RUNS):
            run_detection(detector_info, sample_image)

    print("Warm-up completed.")


# ============================================================
# FACE RESULT INFORMATION
# ============================================================

def extract_face_info(faces):

    count = len(faces)

    confidences = []
    largest_face_area = 0
    largest_bbox = None

    for face in faces:

        confidence = getattr(face, "confidence", None)

        if confidence is not None:
            confidences.append(float(confidence))

        bbox = getattr(face, "bbox", None)

        if bbox is not None:

            x1, y1, x2, y2 = map(float, bbox)

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            area = width * height

            if area > largest_face_area:
                largest_face_area = area
                largest_bbox = [x1, y1, x2, y2]

    max_confidence = (
        max(confidences) if confidences else None
    )

    return {
        "face_count": count,
        "max_confidence": max_confidence,
        "largest_face_area": largest_face_area,
        "largest_bbox": largest_bbox,
    }


# ============================================================
# BENCHMARK ONE DETECTOR
# ============================================================

def benchmark_detector(name, detector_info, image_paths):

    print(f"\n{'=' * 65}")
    print(f"BENCHMARKING: {name}")
    print(f"{'=' * 65}")

    results = []
    latencies = []

    detected_images = 0
    no_face_images = 0
    failed_images = 0
    total_faces = 0

    start_total = time.perf_counter()

    for index, image_path in enumerate(image_paths, start=1):

        image = cv2.imread(str(image_path))

        if image is None:

            failed_images += 1

            results.append({
                "detector": name,
                "image": image_path.name,
                "status": "UNREADABLE",
                "face_count": 0,
                "max_confidence": None,
                "latency_ms": None,
                "largest_face_area": 0,
                "largest_bbox": None,
                "error": "cv2.imread returned None",
            })

            continue

        try:

            # Only inference time is measured.
            inference_start = time.perf_counter()

            faces = run_detection(detector_info, image)

            inference_end = time.perf_counter()

            latency_ms = (
                inference_end - inference_start
            ) * 1000

            latencies.append(latency_ms)

            face_info = extract_face_info(faces)

            face_count = face_info["face_count"]

            total_faces += face_count

            if face_count > 0:
                detected_images += 1
                status = "DETECTED"
            else:
                no_face_images += 1
                status = "NO_FACE"

            results.append({
                "detector": name,
                "image": image_path.name,
                "status": status,
                "face_count": face_count,
                "max_confidence": face_info["max_confidence"],
                "latency_ms": round(latency_ms, 3),
                "largest_face_area": round(
                    face_info["largest_face_area"], 2
                ),
                "largest_bbox": face_info["largest_bbox"],
                "error": None,
            })

        except Exception as exc:

            failed_images += 1

            results.append({
                "detector": name,
                "image": image_path.name,
                "status": "ERROR",
                "face_count": 0,
                "max_confidence": None,
                "latency_ms": None,
                "largest_face_area": 0,
                "largest_bbox": None,
                "error": repr(exc),
            })

        if index % 250 == 0 or index == len(image_paths):

            print(
                f"Progress: {index}/{len(image_paths)} | "
                f"Detected: {detected_images} | "
                f"No face: {no_face_images} | "
                f"Errors: {failed_images}"
            )

    total_time = time.perf_counter() - start_total

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    processed = len(image_paths)

    detection_rate = (
        detected_images / processed * 100
        if processed else 0
    )

    avg_latency = (
        float(np.mean(latencies)) if latencies else 0
    )

    p50_latency = (
        float(np.percentile(latencies, 50))
        if latencies else 0
    )

    p95_latency = (
        float(np.percentile(latencies, 95))
        if latencies else 0
    )

    p99_latency = (
        float(np.percentile(latencies, 99))
        if latencies else 0
    )

    fps = (
        len(latencies) / sum(latencies) * 1000
        if latencies and sum(latencies) > 0 else 0
    )

    summary = {
        "detector": name,
        "total_images": processed,
        "detected_images": detected_images,
        "no_face_images": no_face_images,
        "failed_images": failed_images,
        "total_faces": total_faces,
        "detection_rate_percent": round(detection_rate, 2),
        "avg_latency_ms": round(avg_latency, 3),
        "p50_latency_ms": round(p50_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "p99_latency_ms": round(p99_latency, 3),
        "fps": round(fps, 2),
        "total_runtime_seconds": round(total_time, 2),
    }

    print("\nRESULT:")
    for key, value in summary.items():
        print(f"{key:30s}: {value}")

    return summary, results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--employee-dir",
        type=Path,
        default=EMPLOYEE_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
    )

    args = parser.parse_args()

    employee_dir = args.employee_dir
    output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if not employee_dir.exists():
        raise FileNotFoundError(
            f"Employee directory not found: {employee_dir}"
        )

    image_paths = get_images(employee_dir)

    if not image_paths:
        raise RuntimeError("No images found.")

    print("\n" + "=" * 65)
    print("FACE DETECTOR BENCHMARK")
    print("=" * 65)

    print(f"Employee folder : {employee_dir}")
    print(f"Total images    : {len(image_paths)}")
    print(f"Output folder   : {output_dir}")
    print(f"Image size      : {IMAGE_SIZE}")
    print(f"Confidence      : {CONFIDENCE_THRESHOLD}")
    print(f"NMS threshold   : {NMS_THRESHOLD}")
    print(f"Warm-up runs    : {WARMUP_RUNS}")
    print(f"Providers       : {PROVIDERS}")

    detectors = initialize_detectors()

    # Read a sample image for warm-up.
    sample_image = cv2.imread(str(image_paths[0]))

    if sample_image is None:
        raise RuntimeError(
            f"Cannot read warm-up image: {image_paths[0]}"
        )

    warmup_detectors(detectors, sample_image)

    summaries = []
    all_results = []

    for name, detector_info in detectors.items():

        summary, results = benchmark_detector(
            name,
            detector_info,
            image_paths,
        )

        summaries.append(summary)
        all_results.extend(results)

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    summary_df = pd.DataFrame(summaries)
    results_df = pd.DataFrame(all_results)

    summary_path = output_dir / "benchmark_summary.csv"
    details_path = output_dir / "benchmark_per_image.csv"

    summary_df.to_csv(summary_path, index=False)
    results_df.to_csv(details_path, index=False)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    print(
        summary_df[
            [
                "detector",
                "detected_images",
                "no_face_images",
                "detection_rate_percent",
                "avg_latency_ms",
                "p95_latency_ms",
                "fps",
            ]
        ].to_string(index=False)
    )

    print("\nReports saved:")
    print(summary_path)
    print(details_path)

    # --------------------------------------------------------
    # List images missed by each detector
    # --------------------------------------------------------

    missed = results_df[
        results_df["status"] == "NO_FACE"
    ]

    missed_path = output_dir / "missed_images.csv"
    missed.to_csv(missed_path, index=False)

    print(f"\nMissed image report: {missed_path}")


if __name__ == "__main__":
    main()