# Face Detection & Recognition Benchmarks

A comprehensive research and benchmarking suite evaluating state-of-the-art **Face Detectors** and **Face Recognizers (Feature Extractors)** across server and edge deployment targets (CPU, GPU with CUDA/TensorRT, and Edge devices like Raspberry Pi).

---

## 📑 Overview of Modules & Tested Models

### 1. Face Detectors Benchmarked
* **RetinaFace (MobileNetV2):** Highest recall & detection coverage (**100.0% coverage**, 0 missed faces across 5,187 challenging images at ~120 FPS on GPU).
* **SCRFD (500M / 2.5G / 10G):** Ultra-fast lightweight detector (**~299 FPS / 3.35 ms** on GPU with 99.34% coverage).
* **BlazeFace & YuNet:** Low-latency mobile/edge face detectors.

### 2. Face Recognizers Benchmarked
* **ArcFace (ResNet-50 / W600K):** High-precision 512-D identity embeddings with minimal inter-class collision (mean impostor similarity: **0.0092**; inference latency: **0.88 ms** on GPU).
* **MobileFaceNet (INT8 Quantized TFLite):** Edge-optimized 128-D recognizer (1.54 MB model size, runs without GPU dependencies).
* **AdaFace, EdgeFace, SphereFace:** Accuracy and margin evaluation via the UniFace suite.
* **GhostFaceNet (DeepFace CPU):** Lightweight CPU-targeted embedding model.

---

## 📁 Repository Structure

```
face/
├── uniface/                                # Core benchmark and recognition pipelines
│   ├── yolo_arcface_employee_recognition.py # Real-time recognition pipeline (RetinaFace + ArcFace)
│   ├── benchmark_arcface_vs_mobilefacenet.py # Direct ArcFace vs. MobileFaceNet comparison
│   ├── benchmark_insightface.py            # InsightFace multi-model evaluator
│   ├── benchmark1.py                       # UniFace 5-model recognizer benchmark
│   ├── bench_demo.py                       # Detector latency & coverage comparison
│   ├── camera.py / camera_test2.py         # Live camera tracking & recognition
│   └── compare_detector_misses.py          # Failure analysis between SCRFD & RetinaFace
│
├── mobile_net/                             # MobileFaceNet INT8 TFLite deployment
│   └── check_camera.py                     # Standalone TFLite camera recognition
│
├── blaze_face/                             # MediaPipe BlazeFace detection experiments
│   └── check.py                            # BlazeFace TFLite inference script
│
├── deepface/                               # GhostFaceNet & DeepFace evaluation
│   ├── check.py                            # Live GhostFaceNet camera script
│   └── deepface_test.py                    # GhostFaceNet latency & embedding test
│
├── "yuNet + MobileFaceNe"/                 # YuNet ONNX detector + MobileFaceNet pipeline
│   └── check.py                            # OpenCV YuNet camera inference
│
├── VIT/                                    # Vision Transformer (ViT) architecture study
│   ├── vit_analysis.md                     # ViT classification vs. detection breakdown
│   ├── split_image.py                      # Patch slicing & projection demo
│   └── test.py                             # ViT Hugging Face inference test
│
├── yolo_arcface_results_v4/                # Verification utilities for enrolled databases
│   └── verify_embeddings.py                # Database integrity & margin validator
│
└── make.py                                 # Image dataset preprocessing utility
```

---

## 📊 Benchmark Results

### 1. Detector Comparison (5,187 Images on GPU)
| Detector | Backbone | Detection Rate | Faces Found | Avg Latency | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RetinaFace** | MobileNetV2 | **100.0%** (0 misses) | **5,380** | 8.35 ms | 119.8 FPS |
| **SCRFD** | 500M | 99.34% (34 misses) | 5,217 | **3.35 ms** | **298.9 FPS** ⚡ |
| **SCRFD** | 10G | 99.33% (35 misses) | 5,234 | 7.14 ms | 140.1 FPS |

### 2. Recognizer Comparison (ArcFace vs. MobileFaceNet INT8)
| Recognizer | Embedding Dim | Model Size | Avg Latency | Impostor Mean | False Match Risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ArcFace ONNX (CUDA)** | **512-D** | 129.4 MB | **0.88 ms** | **0.0092** | Extremely low |
| **MobileFaceNet (INT8 TFLite)** | 128-D | **1.54 MB** | 27.68 ms (CPU) | 0.0886 | Moderate on 50k+ faces |

---

## 🚀 Quick Start

### Installation
```bash
conda create -n face python=3.10 -y
conda activate face
pip install opencv-python numpy onnxruntime-gpu uniface ai-edge-litert
```

### Run Live Face Recognition with Pre-built Database
```bash
python uniface/yolo_arcface_employee_recognition.py --camera-only
```

### Validate Embeddings Database Integrity
```bash
python yolo_arcface_results_v4/verify_embeddings.py
```
