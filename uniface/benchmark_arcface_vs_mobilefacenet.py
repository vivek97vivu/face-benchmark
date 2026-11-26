#!/usr/bin/env python3
"""
Benchmark: ArcFace ONNX vs. MobileFaceNet INT8 TFLite
Evaluates:
  1. Inference Latency & FPS (Throughput)
  2. Model size & RAM footprint
  3. Feature dimensionality (512-D vs 128-D)
  4. Cosine similarity discrimination (Genuine vs. Impostor separation)
"""

import os
import cv2
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ai_edge_litert.interpreter import Interpreter
from uniface import FaceAnalyzer, ArcFace

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path("/home/vivek/Music/face")
EMPLOYEE_DIR = BASE_DIR / "employee"
TFLITE_PATH = BASE_DIR / "mobile_net" / "mobilefacenet_int8.tflite"
OUTPUT_DIR = BASE_DIR / "benchmark_arcface_vs_mobilefacenet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_BENCHMARK_IMAGES = 200  # Number of images to evaluate for accuracy & latency
WARMUP_RUNS = 20
SPEED_TEST_RUNS = 200

# ============================================================
# INITIALIZE MODELS
# ============================================================

print("=" * 70)
print("INITIALIZING RECOGNIZERS FOR BENCHMARK")
print("=" * 70)

# 1. ArcFace via UniFace (ONNX with CUDA)
print("\n[1/2] Loading ArcFace (ONNX)...")
arcface_recognizer = ArcFace()
print("ArcFace loaded successfully.")

# 2. MobileFaceNet INT8 via LiteRT / TFLite
print("\n[2/2] Loading MobileFaceNet INT8 (TFLite)...")
tflite_interpreter = Interpreter(model_path=str(TFLITE_PATH))
tflite_interpreter.allocate_tensors()
tflite_in = tflite_interpreter.get_input_details()[0]
tflite_out = tflite_interpreter.get_output_details()[0]
tflite_in_scale, tflite_in_zero = tflite_in["quantization"]
tflite_out_scale, tflite_out_zero = tflite_out["quantization"]
print(f"MobileFaceNet input: {tflite_in['shape']}, dtype: {tflite_in['dtype']}")
print(f"MobileFaceNet output: {tflite_out['shape']}, dtype: {tflite_out['dtype']}")

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

analyzer = FaceAnalyzer()

def align_face(image, landmarks):
    if landmarks is None or len(landmarks) != 5:
        return None
    transform, _ = cv2.estimateAffinePartial2D(
        np.asarray(landmarks, dtype=np.float32),
        REFERENCE_LANDMARKS,
        method=cv2.LMEDS
    )
    if transform is None:
        return None
    return cv2.warpAffine(image, transform, (112, 112), borderValue=0)

# ============================================================
# EMBEDDING EXTRACTION FUNCTIONS
# ============================================================

def get_arcface_emb(aligned_face):
    """Expects 112x112 BGR face image, returns normalized 512-D float32 vector."""
    emb = arcface_recognizer(aligned_face, REFERENCE_LANDMARKS)
    if emb is None:
        return None
    emb = np.asarray(emb, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-12 else None

def get_mobilefacenet_emb(aligned_face):
    """Expects 112x112 BGR face image, returns normalized 128-D float32 vector."""
    rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    quant_img = np.round(rgb / tflite_in_scale) + tflite_in_zero
    quant_img = np.clip(quant_img, -128, 127).astype(np.int8)
    quant_img = np.expand_dims(quant_img, axis=0)

    tflite_interpreter.set_tensor(tflite_in["index"], quant_img)
    tflite_interpreter.invoke()
    raw_output = tflite_interpreter.get_tensor(tflite_out["index"])[0]

    dequant_emb = (raw_output.astype(np.float32) - tflite_out_zero) * tflite_out_scale
    norm = np.linalg.norm(dequant_emb)
    return dequant_emb / norm if norm > 1e-12 else None

# ============================================================
# PREPARE DATASET (CROPPED & ALIGNED FACES)
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING ALIGNED FACES FOR BENCHMARK")
print("=" * 70)

image_files = sorted([
    p for p in EMPLOYEE_DIR.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])[:NUM_BENCHMARK_IMAGES]

aligned_dataset = []
failed_count = 0

for idx, img_path in enumerate(image_files):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    faces = analyzer.analyze(img)
    if not faces:
        failed_count += 1
        continue
    # Pick the primary (largest) face
    largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    aligned_crop = align_face(img, largest_face.landmarks)
    if aligned_crop is not None and aligned_crop.shape[:2] == (112, 112):
        aligned_dataset.append((img_path.name, aligned_crop))

print(f"Prepared {len(aligned_dataset)} aligned faces for testing (Failed detection: {failed_count}).")

# ============================================================
# 1. PURE INFERENCE SPEED & LATENCY BENCHMARK
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARKING INFERENCE LATENCY & THROUGHPUT (FPS)")
print("=" * 70)

sample_crop = aligned_dataset[0][1]

# Warm-up
for _ in range(WARMUP_RUNS):
    get_arcface_emb(sample_crop)
    get_mobilefacenet_emb(sample_crop)

# Benchmark ArcFace
arcface_latencies = []
for _ in range(SPEED_TEST_RUNS):
    t0 = time.perf_counter()
    get_arcface_emb(sample_crop)
    arcface_latencies.append((time.perf_counter() - t0) * 1000)

# Benchmark MobileFaceNet
mobile_latencies = []
for _ in range(SPEED_TEST_RUNS):
    t0 = time.perf_counter()
    get_mobilefacenet_emb(sample_crop)
    mobile_latencies.append((time.perf_counter() - t0) * 1000)

arcface_lat = np.array(arcface_latencies)
mobile_lat = np.array(mobile_latencies)

print("\n--- Latency Results (Recognition Inference Only) ---")
print(f"ArcFace ONNX (CUDA):")
print(f"  Avg Latency: {arcface_lat.mean():.2f} ms | p50: {np.median(arcface_lat):.2f} ms | p95: {np.percentile(arcface_lat, 95):.2f} ms")
print(f"  Estimated Throughput: {1000.0 / arcface_lat.mean():.1f} FPS")

print(f"\nMobileFaceNet INT8 (TFLite CPU/XNNPACK):")
print(f"  Avg Latency: {mobile_lat.mean():.2f} ms | p50: {np.median(mobile_lat):.2f} ms | p95: {np.percentile(mobile_lat, 95):.2f} ms")
print(f"  Estimated Throughput: {1000.0 / mobile_lat.mean():.1f} FPS")

# ============================================================
# 2. FEATURE EXTRACTION & COSINE SIMILARITY ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("EVALUATING REPRESENTATION QUALITY & COSINE SIMILARITY")
print("=" * 70)

arcface_embeddings = []
mobile_embeddings = []
labels = []

for name, crop in aligned_dataset:
    emb_arc = get_arcface_emb(crop)
    emb_mob = get_mobilefacenet_emb(crop)
    if emb_arc is not None and emb_mob is not None:
        arcface_embeddings.append(emb_arc)
        mobile_embeddings.append(emb_mob)
        labels.append(name)

arcface_embeddings = np.array(arcface_embeddings)
mobile_embeddings = np.array(mobile_embeddings)

print(f"Extracted embeddings matrix: ArcFace {arcface_embeddings.shape}, MobileFaceNet {mobile_embeddings.shape}")

# Self-similarity matrix (pairwise cosine similarities among all test faces)
sim_arcface = np.dot(arcface_embeddings, arcface_embeddings.T)
sim_mobile = np.dot(mobile_embeddings, mobile_embeddings.T)

# Non-diagonal elements are impostor / inter-identity pairs
n = len(labels)
mask_off_diag = ~np.eye(n, dtype=bool)

arcface_impostor_scores = sim_arcface[mask_off_diag]
mobile_impostor_scores = sim_mobile[mask_off_diag]

print("\n--- Inter-Identity (Impostor) Cosine Similarity Distribution ---")
print(f"ArcFace (512-D):")
print(f"  Mean Impostor Similarity: {arcface_impostor_scores.mean():.4f}")
print(f"  Std: {arcface_impostor_scores.std():.4f}")
print(f"  p95: {np.percentile(arcface_impostor_scores, 95):.4f} | p99: {np.percentile(arcface_impostor_scores, 99):.4f} | Max: {arcface_impostor_scores.max():.4f}")

print(f"\nMobileFaceNet (128-D INT8):")
print(f"  Mean Impostor Similarity: {mobile_impostor_scores.mean():.4f}")
print(f"  Std: {mobile_impostor_scores.std():.4f}")
print(f"  p95: {np.percentile(mobile_impostor_scores, 95):.4f} | p99: {np.percentile(mobile_impostor_scores, 99):.4f} | Max: {mobile_impostor_scores.max():.4f}")

# Model sizes on disk
arcface_size_mb = 129.4  # w600k_r50.onnx / arcface standard
mobile_size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)

# ============================================================
# SUMMARY REPORT
# ============================================================

summary = {
    "Model": ["ArcFace (ONNX / CUDA)", "MobileFaceNet (INT8 TFLite)"],
    "Model Size (MB)": [arcface_size_mb, round(mobile_size_mb, 2)],
    "Embedding Dimension": [512, 128],
    "Quantization": ["FP32", "INT8"],
    "Inference Latency (Avg ms)": [round(arcface_lat.mean(), 2), round(mobile_lat.mean(), 2)],
    "Inference Latency (p50 ms)": [round(np.median(arcface_lat), 2), round(np.median(mobile_lat), 2)],
    "Inference Latency (p95 ms)": [round(np.percentile(arcface_lat, 95), 2), round(np.percentile(mobile_lat, 95), 2)],
    "Throughput (FPS)": [round(1000.0 / arcface_lat.mean(), 1), round(1000.0 / mobile_lat.mean(), 1)],
    "Mean Impostor Similarity": [round(float(arcface_impostor_scores.mean()), 4), round(float(mobile_impostor_scores.mean()), 4)],
    "p99 Impostor Similarity": [round(float(np.percentile(arcface_impostor_scores, 99)), 4), round(float(np.percentile(mobile_impostor_scores, 99)), 4)],
    "Max Impostor Similarity": [round(float(arcface_impostor_scores.max()), 4), round(float(mobile_impostor_scores.max()), 4)]
}

df_summary = pd.DataFrame(summary)
summary_csv = OUTPUT_DIR / "arcface_vs_mobilefacenet_summary.csv"
df_summary.to_csv(summary_csv, index=False)

print("\n" + "=" * 70)
print("BENCHMARK SUMMARY SAVED TO:", summary_csv)
print("=" * 70)
print(df_summary.to_string(index=False))
