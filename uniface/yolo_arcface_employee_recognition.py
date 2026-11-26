

import os
import csv
import cv2
import time
import argparse
import itertools
from pathlib import Path
from collections import defaultdict

import numpy as np
from uniface import FaceAnalyzer, RetinaFace


# ============================================================
# CONFIGURATION
# ============================================================

EMPLOYEE_DIR = Path("/home/vivek/Music/face/All_Images")

OUTPUT_DIR = Path("/home/vivek/Music/face/yolo_arcface_results_v4")
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
REPORT_DIR = OUTPUT_DIR / "reports"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
ALIGNED_DIR = OUTPUT_DIR / "aligned_faces"

CAMERA_ID = 4

# Minimum face bounding box side in original-image pixels.
MIN_FACE_SIDE_PX = 20

# Quality metrics are recorded. They do not reject faces by default.
MIN_BLUR_VARIANCE = 0.0
MIN_BRIGHTNESS = 0.0
MAX_BRIGHTNESS = 255.0

# Cosine similarity threshold.
RECOGNITION_THRESHOLD = 0.40

# Camera tracking.
RECOGNIZE_EVERY_N_FRAMES = 5
TRACK_MAX_MISSED_FRAMES = 15
TRACK_MAX_CENTER_DISTANCE = 120

# Save aligned crop for every enrolled image.
SAVE_ALIGNED_FACES = True
SAVE_ANNOTATED_ALL = False
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Standard ArcFace 112x112 five-point template.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# ============================================================
# UTILITIES
# ============================================================

def ensure_directories():
    for directory in (EMBEDDING_DIR, REPORT_DIR, ANNOTATED_DIR, ALIGNED_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def parse_employee_identity(filename):
    """Parse EMPLOYEEID_EmployeeName.ext. Handle Actor_XXX format to avoid collision."""
    stem = Path(filename).stem
    if stem.startswith("Actor_"):
        return stem, stem
    elif "_" in stem:
        employee_id, employee_name = stem.split("_", 1)
        return employee_id.strip(), employee_name.strip()
    else:
        return stem, stem


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        return None
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-12:
        return None
    return embedding / norm


def safe_filename(value):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(value))


def face_quality_metrics(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return 0, 0, 0.0, 0.0

    face_w = max(0, x2 - x1)
    face_h = max(0, y2 - y1)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    return face_w, face_h, blur_variance, brightness


# ============================================================
# MODEL INITIALIZATION
# ============================================================

class FacePipeline:
    def __init__(self):
        print("\nLoading FaceAnalyzer with RetinaFace detector + ArcFace recognizer...")
        detector = RetinaFace(confidence_threshold=0.30)
        self.analyzer = FaceAnalyzer(detector=detector)
        print(f"Analyzer initialized: {self.analyzer}")

    def detect(self, image):
        """Analyze image with FaceAnalyzer and return list of detections with bbox, confidence, landmarks, embedding."""
        if image is None or image.size == 0:
            return []

        faces = self.analyzer.analyze(image)
        detections = []
        for face in faces:
            bbox = np.asarray(face.bbox, dtype=np.float32)
            conf = float(face.confidence) if hasattr(face, "confidence") and face.confidence is not None else 1.0
            landmarks = (
                np.asarray(face.landmarks, dtype=np.float32)
                if hasattr(face, "landmarks") and face.landmarks is not None
                else None
            )
            embedding = (
                normalize_embedding(face.embedding)
                if hasattr(face, "embedding") and face.embedding is not None
                else None
            )

            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "landmarks": landmarks,
                "embedding": embedding,
                "face_obj": face,
            })
        return detections

    def align_face(self, image, detection):
        """
        Align using detector's landmarks or fallback to bounding box crop.
        """
        bbox = detection["bbox"]
        landmarks = detection["landmarks"]

        if landmarks is not None and len(landmarks) == 5:
            matrix, _ = cv2.estimateAffinePartial2D(
                landmarks.astype(np.float32),
                ARCFACE_TEMPLATE,
                method=cv2.LMEDS,
            )
            if matrix is not None and np.all(np.isfinite(matrix)):
                aligned = cv2.warpAffine(
                    image,
                    matrix,
                    (112, 112),
                    borderValue=0,
                )
                return aligned, "five_point"

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None, "invalid_bbox"

        aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        return aligned, "bbox_fallback"

    def embed_aligned_face(self, aligned_bgr, detection=None):
        """
        Return embedding. If already extracted in detection dict, use it.
        Otherwise analyze the aligned crop directly.
        """
        if detection is not None and detection.get("embedding") is not None:
            return detection["embedding"]

        if aligned_bgr is None or aligned_bgr.size == 0:
            return None

        crops = self.analyzer.analyze(aligned_bgr)
        if crops and crops[0].embedding is not None:
            return normalize_embedding(crops[0].embedding)
        return None


# ============================================================
# ENROLLMENT AND REPORTING
# ============================================================

def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def annotate_detections(image, detections, selected_index=None):
    annotated = image.copy()
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
        if idx == selected_index:
            color = (0, 255, 0)       # selected largest face
            label = "SELECTED"
        else:
            color = (0, 200, 255)     # other detected faces
            label = "OTHER FACE"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label} {det['confidence']:.2f}",
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return annotated


def enroll_database(pipeline):
    print("\n" + "=" * 72)
    print("EMPLOYEE ENROLLMENT")
    print("=" * 72)

    if not EMPLOYEE_DIR.is_dir():
        raise FileNotFoundError(f"Employee image directory not found: {EMPLOYEE_DIR}")

    files = sorted(
        p for p in EMPLOYEE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"Employee images found: {len(files)}")
    # print(f"Detection confidence: {DETECTION_CONFIDENCE}")
    # print(f"Detection image size: {DETECTION_IMAGE_SIZE}")

    embeddings = []
    employee_ids = []
    employee_names = []
    metadata_rows = []
    image_rows = []

    for index, path in enumerate(files, 1):
        image = cv2.imread(str(path))
        if image is None:
            image_rows.append({
                "image": path.name,
                "employee_id": "",
                "employee_name": "",
                "status": "UNREADABLE",
                "detected_faces": 0,
                "selected_confidence": "",
                "face_width": "",
                "face_height": "",
                "blur_variance": "",
                "brightness": "",
                "alignment": "",
                "note": "cv2.imread returned None",
            })
            print(f"[{index}/{len(files)}] UNREADABLE: {path.name}")
            continue

        emp_id, emp_name = parse_employee_identity(path.name)
        detections = pipeline.detect(image)

        if not detections:
            image_rows.append({
                "image": path.name,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "status": "NO_FACE_DETECTED",
                "detected_faces": 0,
                "selected_confidence": "",
                "face_width": "",
                "face_height": "",
                "blur_variance": "",
                "brightness": "",
                "alignment": "",
                "note": "No face detected by FaceAnalyzer",
            })
            annotated = image.copy()
            cv2.putText(
                annotated, "NO FACE DETECTED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
            )
            cv2.imwrite(str(ANNOTATED_DIR / f"{safe_filename(path.stem)}_NO_FACE.jpg"), annotated)
            print(f"[{index}/{len(files)}] NO FACE: {path.name}")
            continue

        # User-requested rule: use largest bbox as the primary face for this image.
        areas = [
            max(0.0, float(d["bbox"][2] - d["bbox"][0]))
            * max(0.0, float(d["bbox"][3] - d["bbox"][1]))
            for d in detections
        ]
        selected_index = int(np.argmax(areas))
        selected = detections[selected_index]

        face_w, face_h, blur, brightness = face_quality_metrics(
            image, selected["bbox"]
        )

        status = "SUCCESS"
        notes = []

        if len(detections) > 1:
            notes.append("MULTIPLE_FACES_LARGEST_SELECTED")

        if min(face_w, face_h) < MIN_FACE_SIDE_PX:
            status = "FACE_TOO_SMALL"
            notes.append(f"min_side<{MIN_FACE_SIDE_PX}px")

        if blur < MIN_BLUR_VARIANCE:
            status = "LOW_QUALITY_BLUR"
            notes.append("blur_below_configured_minimum")

        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            status = "LOW_QUALITY_BRIGHTNESS"
            notes.append("brightness_outside_configured_range")

        annotated = annotate_detections(image, detections, selected_index)
        if len(detections) > 1 or status != "SUCCESS" or SAVE_ANNOTATED_ALL:
            suffix = "MULTI" if len(detections) > 1 else status
            cv2.imwrite(
                str(ANNOTATED_DIR / f"{safe_filename(path.stem)}_{suffix}.jpg"),
                annotated,
            )

        if status != "SUCCESS":
            image_rows.append({
                "image": path.name,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "status": status,
                "detected_faces": len(detections),
                "selected_confidence": selected["confidence"],
                "face_width": face_w,
                "face_height": face_h,
                "blur_variance": blur,
                "brightness": brightness,
                "alignment": "",
                "note": ";".join(notes),
            })
            print(f"[{index}/{len(files)}] {status}: {path.name}")
            continue

        aligned, alignment_method = pipeline.align_face(image, selected)
        if aligned is None:
            image_rows.append({
                "image": path.name,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "status": "ALIGNMENT_FAILED",
                "detected_faces": len(detections),
                "selected_confidence": selected["confidence"],
                "face_width": face_w,
                "face_height": face_h,
                "blur_variance": blur,
                "brightness": brightness,
                "alignment": alignment_method,
                "note": ";".join(notes),
            })
            print(f"[{index}/{len(files)}] ALIGNMENT FAILED: {path.name}")
            continue

        embedding = pipeline.embed_aligned_face(aligned, detection=selected)
        if embedding is None:
            image_rows.append({
                "image": path.name,
                "employee_id": emp_id,
                "employee_name": emp_name,
                "status": "EMBEDDING_FAILED",
                "detected_faces": len(detections),
                "selected_confidence": selected["confidence"],
                "face_width": face_w,
                "face_height": face_h,
                "blur_variance": blur,
                "brightness": brightness,
                "alignment": alignment_method,
                "note": ";".join(notes),
            })
            print(f"[{index}/{len(files)}] EMBEDDING FAILED: {path.name}")
            continue

        embeddings.append(embedding)
        employee_ids.append(emp_id)
        employee_names.append(emp_name)

        aligned_name = f"{safe_filename(emp_id)}_{safe_filename(path.stem)}_aligned.jpg"
        if SAVE_ALIGNED_FACES:
            cv2.imwrite(str(ALIGNED_DIR / aligned_name), aligned)

        metadata_rows.append({
            "embedding_index": len(embeddings) - 1,
            "employee_id": emp_id,
            "employee_name": emp_name,
            "image": path.name,
            "detection_confidence": selected["confidence"],
            "face_width": face_w,
            "face_height": face_h,
            "blur_variance": blur,
            "brightness": brightness,
            "alignment": alignment_method,
            "detected_faces_in_image": len(detections),
            "aligned_crop": aligned_name if SAVE_ALIGNED_FACES else "",
        })

        image_rows.append({
            "image": path.name,
            "employee_id": emp_id,
            "employee_name": emp_name,
            "status": "SUCCESS",
            "detected_faces": len(detections),
            "selected_confidence": selected["confidence"],
            "face_width": face_w,
            "face_height": face_h,
            "blur_variance": blur,
            "brightness": brightness,
            "alignment": alignment_method,
            "note": ";".join(notes),
        })

        print(
            f"[{index}/{len(files)}] SUCCESS {path.name} | "
            f"faces={len(detections)} | selected={face_w}x{face_h} | "
            f"conf={selected['confidence']:.3f} | align={alignment_method}"
        )

    if not embeddings:
        raise RuntimeError("No valid employee embeddings were generated.")

    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    ids_array = np.asarray(employee_ids, dtype=str)
    names_array = np.asarray(employee_names, dtype=str)

    np.save(EMBEDDING_DIR / "employee_embeddings.npy", embeddings_array)
    np.save(EMBEDDING_DIR / "employee_ids.npy", ids_array)
    np.save(EMBEDDING_DIR / "employee_names.npy", names_array)

    metadata_fields = [
        "embedding_index", "employee_id", "employee_name", "image",
        "detection_confidence", "face_width", "face_height",
        "blur_variance", "brightness", "alignment",
        "detected_faces_in_image", "aligned_crop",
    ]
    image_fields = [
        "image", "employee_id", "employee_name", "status",
        "detected_faces", "selected_confidence", "face_width",
        "face_height", "blur_variance", "brightness", "alignment", "note",
    ]

    save_csv(EMBEDDING_DIR / "embedding_metadata.csv", metadata_rows, metadata_fields)
    save_csv(REPORT_DIR / "image_results.csv", image_rows, image_fields)

    failed_rows = [row for row in image_rows if row["status"] != "SUCCESS"]
    save_csv(REPORT_DIR / "failed_images.csv", failed_rows, image_fields)

    print("\nEnrollment summary")
    print(f"Images scanned: {len(files)}")
    print(f"Embeddings created: {len(embeddings_array)}")
    print(f"Unique employee IDs: {len(set(employee_ids))}")
    print(f"Failed images: {len(failed_rows)}")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Reports: {REPORT_DIR}")

    calibration = calibrate_threshold(embeddings_array, ids_array)
    return embeddings_array, ids_array, names_array, calibration


# ============================================================
# PRELIMINARY THRESHOLD CALIBRATION
# ============================================================

def calibrate_threshold(embeddings, employee_ids, max_pairs=100000, seed=42):
    """
    Estimate genuine/impostor score distributions from enrolled images.

    Genuine pairs: two different images with the same employee ID.
    Impostor pairs: images with different employee IDs.

    This is only a preliminary estimate because enrollment images are not an
    independent validation set. A production threshold must be checked on
    separate camera images.
    """
    print("\n" + "=" * 72)
    print("PRELIMINARY SIMILARITY THRESHOLD CALIBRATION")
    print("=" * 72)

    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for idx, emp_id in enumerate(employee_ids):
        groups[str(emp_id)].append(idx)

    genuine_pairs = []
    for emp_id, indices in groups.items():
        if len(indices) >= 2:
            pairs = list(itertools.combinations(indices, 2))
            if len(pairs) > max_pairs:
                chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
                pairs = [pairs[i] for i in chosen]
            genuine_pairs.extend(pairs)

    ids = np.asarray(employee_ids).astype(str)
    impostor_pairs = []
    n = len(ids)
    attempts = 0
    target_impostor = min(max_pairs, max(1000, len(genuine_pairs) * 3))
    while len(impostor_pairs) < target_impostor and attempts < target_impostor * 20:
        a = int(rng.integers(0, n))
        b = int(rng.integers(0, n))
        if a != b and ids[a] != ids[b]:
            impostor_pairs.append((a, b))
        attempts += 1

    genuine_scores = np.array(
        [float(np.dot(embeddings[a], embeddings[b])) for a, b in genuine_pairs],
        dtype=np.float32,
    )
    impostor_scores = np.array(
        [float(np.dot(embeddings[a], embeddings[b])) for a, b in impostor_pairs],
        dtype=np.float32,
    )

    lines = [
        "PRELIMINARY CALIBRATION - NOT AN INDEPENDENT VALIDATION",
        f"Employees with >=2 enrolled images: {sum(len(v) >= 2 for v in groups.values())}",
        f"Genuine pairs: {len(genuine_scores)}",
        f"Impostor pairs: {len(impostor_scores)}",
    ]

    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        lines.append(
            "Not enough repeated images per employee to estimate both score distributions. "
            "Collect at least two independently captured images per employee."
        )
        summary = "\n".join(lines)
        (REPORT_DIR / "threshold_calibration.txt").write_text(summary, encoding="utf-8")
        print(summary)
        return None

    lines.append(
        "Genuine scores: "
        f"mean={genuine_scores.mean():.4f}, "
        f"p05={np.percentile(genuine_scores, 5):.4f}, "
        f"p50={np.percentile(genuine_scores, 50):.4f}, "
        f"p95={np.percentile(genuine_scores, 95):.4f}"
    )
    lines.append(
        "Impostor scores: "
        f"mean={impostor_scores.mean():.4f}, "
        f"p95={np.percentile(impostor_scores, 95):.4f}, "
        f"p99={np.percentile(impostor_scores, 99):.4f}, "
        f"max={impostor_scores.max():.4f}"
    )

    best = None
    for threshold in np.arange(0.20, 0.901, 0.005):
        false_reject_rate = float(np.mean(genuine_scores < threshold))
        false_accept_rate = float(np.mean(impostor_scores >= threshold))
        balanced_accuracy = 1.0 - (false_reject_rate + false_accept_rate) / 2.0
        candidate = (
            balanced_accuracy,
            float(threshold),
            false_accept_rate,
            false_reject_rate,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate

    lines.append(
        "Best sampled balanced-accuracy threshold: "
        f"{best[1]:.3f} | FAR={best[2]:.4%} | FRR={best[3]:.4%} | "
        f"balanced_accuracy={best[0]:.4%}"
    )
    lines.append(
        "Use this only as a starting point. Validate with separate live-camera "
        "genuine and impostor samples before production."
    )

    summary = "\n".join(lines)
    (REPORT_DIR / "threshold_calibration.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return best[1]


# ============================================================
# SIMPLE CENTROID TRACKER
# ============================================================

class CentroidTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def center(bbox):
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def update(self, detections):
        for track in self.tracks.values():
            track["missed"] += 1

        unmatched = set(range(len(detections)))
        assignments = {}

        candidates = []
        for track_id, track in self.tracks.items():
            old_center = track["center"]
            for det_idx, det in enumerate(detections):
                new_center = self.center(det["bbox"])
                distance = float(np.linalg.norm(old_center - new_center))
                if distance <= TRACK_MAX_CENTER_DISTANCE:
                    candidates.append((distance, track_id, det_idx))

        used_tracks = set()
        used_detections = set()
        for _, track_id, det_idx in sorted(candidates):
            if track_id in used_tracks or det_idx in used_detections:
                continue
            used_tracks.add(track_id)
            used_detections.add(det_idx)
            assignments[det_idx] = track_id
            unmatched.discard(det_idx)

        for det_idx, track_id in assignments.items():
            det = detections[det_idx]
            self.tracks[track_id]["center"] = self.center(det["bbox"])
            self.tracks[track_id]["bbox"] = det["bbox"]
            self.tracks[track_id]["missed"] = 0

        for det_idx in sorted(unmatched):
            det = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = {
                "center": self.center(det["bbox"]),
                "bbox": det["bbox"],
                "missed": 0,
                "label": "UNRECOGNIZED",
                "employee_id": "",
                "employee_name": "",
                "score": 0.0,
                "last_recognition_frame": -RECOGNIZE_EVERY_N_FRAMES,
            }
            assignments[det_idx] = track_id

        expired = [
            track_id for track_id, track in self.tracks.items()
            if track["missed"] > TRACK_MAX_MISSED_FRAMES
        ]
        for track_id in expired:
            del self.tracks[track_id]

        return assignments


# ============================================================
# LIVE CAMERA RECOGNITION
# ============================================================

def run_camera(pipeline, embeddings, employee_ids, employee_names):
    print("\n" + "=" * 72)
    print("LIVE CAMERA RECOGNITION")
    print("=" * 72)

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera ID {CAMERA_ID}")

    tracker = CentroidTracker()
    frame_number = 0
    fps = 0.0
    fps_start = time.time()
    fps_frames = 0

    print("Camera started. Press Q to quit.")
    print(f"Recognition threshold: {RECOGNITION_THRESHOLD:.3f}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            frame_number += 1
            fps_frames += 1

            detections = pipeline.detect(frame)
            assignments = tracker.update(detections)

            for det_idx, det in enumerate(detections):
                track_id = assignments[det_idx]
                track = tracker.tracks[track_id]

                should_recognize = (
                    frame_number - track["last_recognition_frame"]
                    >= RECOGNIZE_EVERY_N_FRAMES
                    or track["label"] == "UNRECOGNIZED"
                )

                if should_recognize:
                    aligned, alignment_method = pipeline.align_face(frame, det)
                    embedding = pipeline.embed_aligned_face(aligned, detection=det)

                    track["last_recognition_frame"] = frame_number

                    if embedding is not None:
                        scores = embeddings @ embedding
                        best_index = int(np.argmax(scores))
                        best_score = float(scores[best_index])

                        track["score"] = best_score
                        if best_score >= RECOGNITION_THRESHOLD:
                            track["label"] = "KNOWN"
                            track["employee_id"] = str(employee_ids[best_index])
                            track["employee_name"] = str(employee_names[best_index])
                        else:
                            track["label"] = "UNKNOWN"
                            track["employee_id"] = ""
                            track["employee_name"] = ""

                        print(
                            f"Track {track_id:<4} {track['label']:<7} "
                            f"{track['employee_id']:<12} "
                            f"{track['employee_name']:<25} "
                            f"score={track['score']:.4f} "
                            f"det_conf={det['confidence']:.3f}"
                        )

                x1, y1, x2, y2 = [int(round(v)) for v in det["bbox"]]
                is_known = track["label"] == "KNOWN"
                color = (0, 255, 0) if is_known else (0, 0, 255)

                label = (
                    f"{track['employee_id']} {track['employee_name']}"
                    if is_known else "UNKNOWN"
                )
                detail = (
                    f"T{track_id} {track['score']:.3f} "
                    f"det:{det['confidence']:.2f}"
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, label, (x1, max(22, y1 - 28)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                )
                cv2.putText(
                    frame, detail, (x1, max(42, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2
                )

            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_start = time.time()

            cv2.putText(
                frame, f"FPS: {fps:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"ArcFace threshold: {RECOGNITION_THRESHOLD:.2f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

            cv2.imshow("FaceAnalyzer Employee Recognition", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Camera stopped.")


# ============================================================
# MAIN
# ============================================================

def load_database():
    emb_path = EMBEDDING_DIR / "employee_embeddings.npy"
    ids_path = EMBEDDING_DIR / "employee_ids.npy"
    names_path = EMBEDDING_DIR / "employee_names.npy"

    if not (emb_path.exists() and ids_path.exists() and names_path.exists()):
        raise FileNotFoundError(
            "Saved employee database not found. Run enrollment first."
        )

    embeddings = np.load(emb_path).astype(np.float32)
    ids = np.load(ids_path, allow_pickle=False).astype(str)
    names = np.load(names_path, allow_pickle=False).astype(str)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Expected embeddings shape (N,512), got {embeddings.shape}")
    if len(embeddings) != len(ids) or len(ids) != len(names):
        raise ValueError("Embedding, employee ID and employee name counts do not match.")

    return embeddings, ids, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enroll-only",
        action="store_true",
        help="Build/update the employee embedding database and exit.",
    )
    parser.add_argument(
        "--camera-only",
        action="store_true",
        help="Use the previously saved employee database without re-enrolling.",
    )
    args = parser.parse_args()

    if args.enroll_only and args.camera_only:
        parser.error("Choose either --enroll-only or --camera-only, not both.")

    ensure_directories()
    pipeline = FacePipeline()

    if args.camera_only:
        embeddings, employee_ids, employee_names = load_database()
        print(f"Loaded database: {embeddings.shape}")
    else:
        embeddings, employee_ids, employee_names, _ = enroll_database(pipeline)

    if args.enroll_only:
        print("\nEnrollment complete. Camera was not started.")
        return

    run_camera(pipeline, embeddings, employee_ids, employee_names)


if __name__ == "__main__":
    main()
