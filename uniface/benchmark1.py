import os
import cv2
import time
import json
import pickle
import hashlib
import psutil
import numpy as np
from pathlib import Path
from collections import defaultdict
import csv
import threading
import subprocess
from datetime import datetime

import uniface


# ============================================================
# CONFIGURATION
# ============================================================

# CHANGE THIS TO YOUR EMPLOYEE FOLDER
EMPLOYEE_DIR = Path("/home/vivek/Music/face/employee")
# Camera ID
CAMERA_ID = 4

# Database/cache directory
DB_DIR = Path("./benchmark_db")

# Results directory
RESULTS_DIR = Path("./benchmark_results")

# Recognition threshold
# Start with this value; later we can calculate the optimum
# threshold from genuine/impostor pairs.
THRESHOLD = 0.4

# Detection confidence
DET_THRESHOLD = 0.4

# # Number of warm-up inference calls
# WARMUP_RUNS = 10

# # Number of benchmark inference measurements
# BENCHMARK_RUNS = 100

# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}
# ============================================================
# RESOURCE MONITOR
# ============================================================

RESOURCE_INTERVAL = 0.5  # seconds


class ResourceMonitor:

    def __init__(self, output_dir):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.csv_path = (
            self.output_dir /
            "resource_usage.csv"
        )

        self.running = False
        self.thread = None

        self.rows = []

        self.process = psutil.Process(
            os.getpid()
        )

        # Initialize CPU measurement
        self.process.cpu_percent(
            interval=None
        )

        psutil.cpu_percent(
            interval=None
        )

    def get_gpu_info(self):

        try:

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu="
                    "utilization.gpu,"
                    "memory.used,"
                    "memory.total,"
                    "temperature.gpu,"
                    "power.draw",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=1
            )

            if result.returncode != 0:
                return {
                    "gpu_util": None,
                    "gpu_memory_used": None,
                    "gpu_memory_total": None,
                    "gpu_memory_percent": None,
                    "gpu_temperature": None,
                    "gpu_power": None,
                }

            line = result.stdout.strip().splitlines()[0]

            parts = [
                x.strip()
                for x in line.split(",")
            ]

            gpu_util = float(parts[0])
            memory_used = float(parts[1])
            memory_total = float(parts[2])
            temperature = float(parts[3])

            try:
                power = float(parts[4])
            except Exception:
                power = None

            memory_percent = (
                memory_used /
                memory_total *
                100
                if memory_total > 0
                else 0
            )

            return {
                "gpu_util": gpu_util,
                "gpu_memory_used": memory_used,
                "gpu_memory_total": memory_total,
                "gpu_memory_percent": memory_percent,
                "gpu_temperature": temperature,
                "gpu_power": power,
            }

        except Exception:

            return {
                "gpu_util": None,
                "gpu_memory_used": None,
                "gpu_memory_total": None,
                "gpu_memory_percent": None,
                "gpu_temperature": None,
                "gpu_power": None,
            }

    def collect(self):

        timestamp = datetime.now().isoformat(
            timespec="milliseconds"
        )

        # --------------------------------------------
        # SYSTEM CPU
        # --------------------------------------------

        system_cpu = psutil.cpu_percent(
            interval=None
        )

        # --------------------------------------------
        # PROCESS CPU
        # --------------------------------------------

        process_cpu = (
            self.process.cpu_percent(
                interval=None
            )
        )

        # --------------------------------------------
        # RAM
        # --------------------------------------------

        memory = psutil.virtual_memory()

        # --------------------------------------------
        # GPU
        # --------------------------------------------

        gpu = self.get_gpu_info()

        row = {
            "timestamp": timestamp,

            "system_cpu_percent":
                system_cpu,

            "process_cpu_percent":
                process_cpu,

            "ram_percent":
                memory.percent,

            "ram_used_mb":
                memory.used /
                (1024 ** 2),

            "gpu_util_percent":
                gpu["gpu_util"],

            "gpu_memory_used_mb":
                gpu["gpu_memory_used"],

            "gpu_memory_total_mb":
                gpu["gpu_memory_total"],

            "gpu_memory_percent":
                gpu["gpu_memory_percent"],

            "gpu_temperature_c":
                gpu["gpu_temperature"],

            "gpu_power_w":
                gpu["gpu_power"],
        }

        self.rows.append(row)

    def monitor_loop(self):

        while self.running:

            self.collect()

            time.sleep(
                RESOURCE_INTERVAL
            )

    def start(self):

        print()
        print("=" * 70)
        print("RESOURCE MONITOR STARTED")
        print("=" * 70)

        self.running = True

        self.thread = threading.Thread(
            target=self.monitor_loop,
            daemon=True
        )

        self.thread.start()

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2
            )

        # One final measurement
        self.collect()

        self.save()

        print()
        print("=" * 70)
        print("RESOURCE MONITOR STOPPED")
        print("=" * 70)

        print(
            f"Resource log: "
            f"{self.csv_path}"
        )

    def save(self):

        if not self.rows:
            return

        fieldnames = list(
            self.rows[0].keys()
        )

        with open(
            self.csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()

            writer.writerows(
                self.rows
            )

    def summary(self):

        if not self.rows:
            return {}

        def values(key):

            return [
                float(row[key])
                for row in self.rows
                if row[key] is not None
            ]

        def stats(key):

            vals = values(key)

            if not vals:
                return {
                    "average": None,
                    "maximum": None,
                    "minimum": None,
                }

            return {
                "average": float(
                    np.mean(vals)
                ),
                "maximum": float(
                    np.max(vals)
                ),
                "minimum": float(
                    np.min(vals)
                ),
            }

        return {
            "system_cpu_percent":
                stats(
                    "system_cpu_percent"
                ),

            "process_cpu_percent":
                stats(
                    "process_cpu_percent"
                ),

            "ram_percent":
                stats(
                    "ram_percent"
                ),

            "gpu_util_percent":
                stats(
                    "gpu_util_percent"
                ),

            "gpu_memory_percent":
                stats(
                    "gpu_memory_percent"
                ),

            "gpu_memory_used_mb":
                stats(
                    "gpu_memory_used_mb"
                ),

            "gpu_temperature_c":
                stats(
                    "gpu_temperature_c"
                ),

            "gpu_power_w":
                stats(
                    "gpu_power_w"
                ),
        }
# ============================================================
# MODEL DEFINITIONS
# ============================================================

MODELS = {
    "AdaFace": uniface.AdaFace,
    "ArcFace": uniface.ArcFace,
    "EdgeFace": uniface.EdgeFace,
    "MobileFace": uniface.MobileFace,
    "SphereFace": uniface.SphereFace,
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

DB_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# UTILITIES
# ============================================================

def file_hash(path, chunk_size=1024 * 1024):
    """
    Calculate SHA256 hash of an image.
    Used to detect new/modified images.
    """

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size)

            if not data:
                break

            h.update(data)

    return h.hexdigest()


def get_employee_images():
    """
    Scan employee images directly from:

        /home/vivek/Music/face/employee/

    Filename format:

        EMPLOYEE_ID_EMPLOYEE_NAME.jpg

    Example:

        10000_khurshed_alam.png
        10001_Prasanth Kuniyil.jpg
        20000_vivek.jpg

    Employee identity is derived from the filename.
    """

    if not EMPLOYEE_DIR.exists():
        raise FileNotFoundError(
            f"Employee directory does not exist:\n{EMPLOYEE_DIR}"
        )

    records = []

    for path in sorted(EMPLOYEE_DIR.iterdir()):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        filename = path.stem

        # Split only at the first "_"
        if "_" in filename:
            employee_id, employee_name = filename.split(
                "_",
                1
            )
        else:
            employee_id = filename
            employee_name = filename

        employee_name = employee_name.replace(
            "_",
            " "
        ).strip()

        records.append({
            "employee_id": employee_id,
            "employee": employee_name,
            "path": str(path.resolve()),
        })

    return records


# ============================================================
# CACHE / MANIFEST
# ============================================================

def manifest_path(model_name):
    return DB_DIR / f"{model_name.lower()}_manifest.json"


def embedding_path(model_name):
    return DB_DIR / f"{model_name.lower()}_embeddings.pkl"


def load_manifest(model_name):

    path = manifest_path(model_name)

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def save_manifest(model_name, manifest):

    with open(manifest_path(model_name), "w") as f:
        json.dump(
            manifest,
            f,
            indent=2
        )


def load_embeddings(model_name):

    path = embedding_path(model_name)

    if not path.exists():
        return {}

    with open(path, "rb") as f:
        return pickle.load(f)


def save_embeddings(model_name, embeddings):

    with open(embedding_path(model_name), "wb") as f:
        pickle.dump(
            embeddings,
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )


# ============================================================
# RECOGNIZER
# ============================================================

def create_recognizer(model_name):

    cls = MODELS[model_name]

    print(f"Loading {model_name}...")

    start = time.perf_counter()

    recognizer = cls()

    elapsed = time.perf_counter() - start

    print(
        f"{model_name} loaded in "
        f"{elapsed:.3f} sec"
    )

    return recognizer, elapsed


# ============================================================
# FACE DETECTOR
# ============================================================

def create_detector():

    print("\nLoading SCRFD detector...")

    start = time.perf_counter()

    detector = uniface.SCRFD()

    elapsed = time.perf_counter() - start

    print(
        f"SCRFD loaded in "
        f"{elapsed:.3f} sec"
    )

    return detector


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(detector, image):
    """
    Detect the largest face.

    IMPORTANT:
    Return the ORIGINAL image and the landmarks.
    Do NOT crop the image because SCRFD landmarks
    are in original-image coordinates and Uniface
    performs the alignment internally.
    """

    try:
        faces = detector.detect(
            image,
            max_num=0,
            metric="max"
        )
    except Exception as e:
        print(f"Detection error: {e}")
        return None

    if not faces:
        return None

    # Largest detected face
    largest = max(
        faces,
        key=lambda f: (
            max(0, f.bbox[2] - f.bbox[0])
            * max(0, f.bbox[3] - f.bbox[1])
        )
    )

    return image, largest


# ============================================================
# EMBEDDING
# ============================================================

def get_embedding(
    recognizer,
    image,
    face
):
    """
    Generate a normalized 512-D embedding.

    Uniface performs face alignment internally using
    the 5-point landmarks.
    """

    embedding = recognizer.get_normalized_embedding(
        image,
        face.landmarks
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    ).reshape(-1)

    return embedding

# ============================================================
# BUILD / UPDATE EMBEDDING DATABASE
# ============================================================

def update_database(
    model_name,
    recognizer,
    detector,
    image_records
):

    print()
    print("=" * 70)
    print(
        f"BUILDING / UPDATING {model_name} DATABASE"
    )
    print("=" * 70)

    old_manifest = load_manifest(
        model_name
    )

    old_embeddings = load_embeddings(
        model_name
    )

    new_manifest = {}

    new_embeddings = {}

    changed = 0
    reused = 0
    failed = 0

    build_start = time.perf_counter()

    for index, record in enumerate(
        image_records,
        start=1
    ):

        employee = record["employee"]
        path = record["path"]

        key = path

        try:
            image_hash = file_hash(path)

        except Exception as e:

            print(
                f"[HASH ERROR] {path}: {e}"
            )

            failed += 1
            continue

        old_entry = old_manifest.get(key)

        # ----------------------------------------------------
        # CACHE HIT
        # ----------------------------------------------------

        if (
            old_entry is not None
            and old_entry.get("hash") == image_hash
            and key in old_embeddings
        ):

            new_manifest[key] = old_entry

            new_embeddings[key] = old_embeddings[key]

            reused += 1

            print(
                f"[{index}/{len(image_records)}] "
                f"REUSE  {employee} / "
                f"{Path(path).name}"
            )

            continue

        # ----------------------------------------------------
        # NEW / MODIFIED IMAGE
        # ----------------------------------------------------

        image = cv2.imread(path)

        if image is None:

            print(
                f"[IMAGE ERROR] {path}"
            )

            failed += 1
            continue

        detection = detect_face(
            detector,
            image
        )

        if detection is None:

            print(
                f"[NO FACE] {employee} / "
                f"{Path(path).name}"
            )

            failed += 1
            continue

        face_image, face = detection

        try:

            embedding = get_embedding(
                recognizer,
                face_image,
                face
            )

        except Exception as e:

            print(
                f"[EMBED ERROR] "
                f"{employee} / "
                f"{Path(path).name}: {e}"
            )

            failed += 1
            continue

        new_manifest[key] = {
            "employee_id": record["employee_id"],
            "employee": employee,
            "hash": image_hash,
        }
        new_embeddings[key] = {
            "employee_id": record["employee_id"],
            "employee": employee,
            "embedding": embedding,
        }

        changed += 1

        print(
            f"[{index}/{len(image_records)}] "
            f"NEW/UPDATED  {employee} / "
            f"{Path(path).name}"
        )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_manifest(
        model_name,
        new_manifest
    )

    save_embeddings(
        model_name,
        new_embeddings
    )

    build_time = (
        time.perf_counter()
        - build_start
    )

    print()
    print(
        f"{model_name} database completed"
    )

    print(
        f"New/updated : {changed}"
    )

    print(
        f"Reused      : {reused}"
    )

    print(
        f"Failed      : {failed}"
    )

    print(
        f"Total       : {len(new_embeddings)}"
    )

    print(
        f"Build time  : {build_time:.3f} sec"
    )

    return {
        "embeddings": new_embeddings,
        "build_time": build_time,
        "changed": changed,
        "reused": reused,
        "failed": failed,
    }


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


# ============================================================
# RECOGNITION
# ============================================================

def recognize(
    query_embedding,
    database
):

    best_name = "Unknown"
    best_score = -1.0

    for item in database.values():

        employee = item["employee"]

        embedding = item["embedding"]

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        if score > best_score:

            best_score = score
            best_name = employee

    if best_score < THRESHOLD:

        return (
            "Unknown",
            best_score
        )

    return (
        best_name,
        best_score
    )


# ============================================================
# CAMERA BENCHMARK
# ============================================================

def run_camera(
    recognizers,
    databases,
    detector
):

    print()
    print("=" * 70)
    print(
        f"OPENING CAMERA {CAMERA_ID}"
    )
    print("=" * 70)

    cap = cv2.VideoCapture(
        CAMERA_ID
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open camera {CAMERA_ID}"
        )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "camera_results.csv"
    )

    csv_file = open(
        csv_path,
        "w",
        buffering=1
    )

    csv_file.write(
        "timestamp,model,"
        "identity,similarity,"
        "inference_ms,total_ms,fps\n"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = {}

    for model_name in MODELS:

        metrics[model_name] = {
            "inference_times": [],
            "total_times": [],
            "similarities": [],
            "frames": 0,
            "recognized": 0,
        }

    frame_counter = 0

    fps_start = time.perf_counter()

    fps = 0.0

    print()
    print(
        "Camera started."
    )

    print(
        "Press Q to quit."
    )

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "Camera frame failed."
            )

            break

        frame_counter += 1

        # ----------------------------------------------------
        # DETECT FACE
        # ----------------------------------------------------

        detection_start = time.perf_counter()

        detection = detect_face(
            detector,
            frame
        )

        detection_time = (
            time.perf_counter()
            - detection_start
        )

        if detection is None:

            cv2.imshow(
                "Uniface Benchmark",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            continue

        face_image, face = detection

        # ----------------------------------------------------
        # MODEL LOOP
        # ----------------------------------------------------

        display_lines = []

        for model_name in MODELS:

            recognizer = recognizers[
                model_name
            ]

            database = databases[
                model_name
            ]

            total_start = time.perf_counter()

            embedding_start = (
                time.perf_counter()
            )

            try:

                embedding = get_embedding(
                    recognizer,
                    frame,
                    face
                )

            except Exception as e:

                print(
                    f"{model_name} error: {e}"
                )

                continue

            inference_time = (
                time.perf_counter()
                - embedding_start
            )

            identity, similarity = recognize(
                embedding,
                database
            )

            total_time = (
                time.perf_counter()
                - total_start
            )

            metrics[
                model_name
            ][
                "inference_times"
            ].append(
                inference_time * 1000
            )

            metrics[
                model_name
            ][
                "total_times"
            ].append(
                (
                    total_time
                    + detection_time
                ) * 1000
            )

            metrics[
                model_name
            ][
                "similarities"
            ].append(
                similarity
            )

            metrics[
                model_name
            ][
                "frames"
            ] += 1

            if identity != "Unknown":

                metrics[
                    model_name
                ][
                    "recognized"
                ] += 1

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            csv_file.write(
                f"{time.time()},"
                f"{model_name},"
                f"{identity},"
                f"{similarity:.6f},"
                f"{inference_time * 1000:.3f},"
                f"{(total_time + detection_time) * 1000:.3f},"
                f"{fps:.2f}\n"
            )

            display_lines.append(
                (
                    model_name,
                    identity,
                    similarity,
                    inference_time * 1000
                )
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - fps_start
        )

        if elapsed >= 1.0:

            fps = (
                frame_counter
                / elapsed
            )

            frame_counter = 0
            fps_start = time.perf_counter()

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        y = 30

        cv2.putText(
            frame,
            f"Camera {CAMERA_ID} | FPS: {fps:.1f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        y += 30

        for (
            model_name,
            identity,
            similarity,
            inference_ms
        ) in display_lines:

            text = (
                f"{model_name}: "
                f"{identity} "
                f"{similarity:.3f} "
                f"{inference_ms:.1f}ms"
            )

            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            y += 25

        cv2.imshow(
            "Uniface Benchmark",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    cap.release()

    csv_file.close()

    cv2.destroyAllWindows()

    return metrics


# ============================================================
# BENCHMARK SUMMARY
# ============================================================

def create_summary(metrics,resource_summary=None):

    print()
    print("=" * 100)
    print(
        "FINAL UNIFACE BENCHMARK"
    )
    print("=" * 100)

    print(
        f"{'MODEL':<15}"
        f"{'AVG(ms)':>12}"
        f"{'P50(ms)':>12}"
        f"{'P95(ms)':>12}"
        f"{'FPS':>12}"
        f"{'RECOG_RATE':>12}"
        f"{'SIM':>12}"
    )

    print("-" * 100)

    summary = {}

    for model_name, data in metrics.items():

        inference_times = np.array(
            data["inference_times"]
        )

        total_times = np.array(
            data["total_times"]
        )

        similarities = np.array(
            data["similarities"]
        )

        if len(inference_times) == 0:

            continue

        avg_ms = float(
            np.mean(inference_times)
        )

        p50_ms = float(
            np.percentile(
                inference_times,
                50
            )
        )

        p95_ms = float(
            np.percentile(
                inference_times,
                95
            )
        )

        fps_value = (
            1000.0 / avg_ms
            if avg_ms > 0
            else 0
        )

        recognition_rate = (
            data["recognized"]
            / data["frames"]
            * 100
            if data["frames"] > 0
            else 0
        )

        mean_similarity = float(
            np.mean(similarities)
        )

        print(
            f"{model_name:<15}"
            f"{avg_ms:>12.3f}"
            f"{p50_ms:>12.3f}"
            f"{p95_ms:>12.3f}"
            f"{fps_value:>12.2f}"
            f"{recognition_rate:>11.2f}%"
            f"{mean_similarity:>12.4f}"
        )

        summary[model_name] = {
            "avg_inference_ms": avg_ms,
            "p50_inference_ms": p50_ms,
            "p95_inference_ms": p95_ms,
            "estimated_embedding_fps": fps_value,
            "recognition_rate": recognition_rate,
            "mean_similarity": mean_similarity,
            "frames": data["frames"],
        }

    print("-" * 100)

    summary_path = (
        RESULTS_DIR /
        "summary.json"
    )

    final_summary = {
        "models": summary,
        "resources": resource_summary or {},
    }

    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            final_summary,
            f,
            indent=2
        )

    print()
    print(
        f"Summary saved to: "
        f"{summary_path}"
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

def system_info():

    vm = psutil.virtual_memory()

    info = {
        "cpu_count": os.cpu_count(),
        "ram_total_gb":
            vm.total / (1024 ** 3),
        "ram_available_gb":
            vm.available / (1024 ** 3),
    }

    try:

        import onnxruntime as ort

        info["onnxruntime_providers"] = (
            ort.get_available_providers()
        )

    except Exception:

        info[
            "onnxruntime_providers"
        ] = []

    print()
    print("=" * 70)
    print("SYSTEM")
    print("=" * 70)

    for key, value in info.items():

        print(
            f"{key}: {value}"
        )

    with open(
        RESULTS_DIR / "system.json",
        "w"
    ) as f:

        json.dump(
            info,
            f,
            indent=2
        )


# ============================================================
# MAIN
# ============================================================

def main():
    # --------------------------------------------------------
    # SYSTEM
    # --------------------------------------------------------

    system_info()

    # --------------------------------------------------------
    # EMPLOYEE IMAGES
    # --------------------------------------------------------

    image_records = (
        get_employee_images()
    )

    print()
    print(
        f"Employee images found: "
        f"{len(image_records)}"
    )

    if not image_records:

        raise RuntimeError(
            "No employee images found."
        )

    employees = sorted(
        set(
            r["employee_id"]
            for r in image_records
        )
    )

    print(
        f"Employees found: "
        f"{len(employees)}"
    )

    # --------------------------------------------------------
    # DETECTOR
    # --------------------------------------------------------

    detector = create_detector()

    # --------------------------------------------------------
    # LOAD RECOGNIZERS
    # --------------------------------------------------------

    recognizers = {}
    load_times = {}

    for model_name in MODELS:

        recognizer, load_time = (
            create_recognizer(
                model_name
            )
        )

        recognizers[
            model_name
        ] = recognizer

        load_times[
            model_name
        ] = load_time

    # --------------------------------------------------------
    # BUILD / UPDATE DATABASES
    # --------------------------------------------------------

    databases = {}
    database_stats = {}

    for model_name in MODELS:

        result = update_database(
            model_name,
            recognizers[model_name],
            detector,
            image_records
        )

        databases[
            model_name
        ] = result["embeddings"]

        database_stats[
            model_name
        ] = {
            "build_time":
                result["build_time"],
            "new_or_updated":
                result["changed"],
            "reused":
                result["reused"],
            "failed":
                result["failed"],
            "total":
                len(result["embeddings"]),
        }

    # --------------------------------------------------------
    # SAVE MODEL INFO
    # --------------------------------------------------------

    with open(
        RESULTS_DIR / "model_info.json",
        "w"
    ) as f:

        json.dump(
            {
                "load_times": load_times,
                "database_stats":
                    database_stats,
                "threshold":
                    THRESHOLD,
                "camera_id":
                    CAMERA_ID,
            },
            f,
            indent=2
        )

    # --------------------------------------------------------
    # START RESOURCE MONITOR
    # ONLY FOR CAMERA BENCHMARK
    # --------------------------------------------------------

    resource_monitor = ResourceMonitor(
        RESULTS_DIR
    )

    resource_monitor.start()

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    metrics = run_camera(
        recognizers,
        databases,
        detector
    )

    # --------------------------------------------------------
    # STOP RESOURCE MONITOR
    # --------------------------------------------------------

    resource_monitor.stop()

    resource_summary = (
        resource_monitor.summary()
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    create_summary(
        metrics,
        resource_summary
    )

    print()
    print("=" * 70)
    print("BENCHMARK FINISHED")
    print("=" * 70)

    print()
    print(
        "Results directory:"
    )

    print(
        RESULTS_DIR.resolve()
    )


if __name__ == "__main__":

    main()