import os
import cv2
import numpy as np
from uniface import FaceAnalyzer


# ============================================================
# CONFIGURATION
# ============================================================

EMPLOYEE_DIR = "/home/vivek/Music/face/employee"
TEST_DIR = ""

# Start with this value only for testing.
# We will tune it later using your actual data.
THRESHOLD = 0.10


# ============================================================
# INITIALIZE MODEL
# ============================================================

print("=" * 60)
print("Initializing Face Recognition Model")
print("=" * 60)

analyzer = FaceAnalyzer()

print("Model loaded.\n")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_embedding(image):
    """
    Detect face and return the embedding.

    Returns:
        embedding, face
    """

    faces = analyzer.analyze(image)

    if len(faces) == 0:
        return None, None

    # If multiple faces are present, use the largest face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) *
                       (f.bbox[3] - f.bbox[1])
    )

    if face.embedding is None:
        return None, None

    embedding = np.asarray(
        face.embedding,
        dtype=np.float32
    )

    # Normalize embedding
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return None, None

    embedding = embedding / norm

    return embedding, face


def cosine_similarity(a, b):
    """
    Calculate cosine similarity.
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return float(np.dot(a, b))


# ============================================================
# LOAD EMPLOYEE DATABASE
# ============================================================

print("=" * 60)
print("Creating Employee Database")
print("=" * 60)

employee_embeddings = []
employee_names = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

employee_files = [
    f for f in os.listdir(EMPLOYEE_DIR)
    if f.lower().endswith(valid_extensions)
]

if len(employee_files) == 0:
    print("ERROR: No employee images found.")
    print(f"Put employee images inside: {EMPLOYEE_DIR}/")
    exit()


for filename in sorted(employee_files):

    image_path = os.path.join(
        EMPLOYEE_DIR,
        filename
    )

    # Name comes from filename
    name = os.path.splitext(filename)[0]

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot read: {filename}")
        continue

    embedding, face = get_embedding(image)

    if embedding is None:
        print(f"[NO FACE] {filename}")
        continue

    employee_embeddings.append(embedding)
    employee_names.append(name)

    print(
        f"[OK] {name:<20} "
        f"Embedding: {embedding.shape}"
    )


if len(employee_embeddings) == 0:
    print("\nERROR: No valid employee faces found.")
    exit()


print("\nEmployee database created.")
print(f"Total employees: {len(employee_names)}")


# ============================================================
# TEST IMAGES
# ============================================================

print("\n")
print("=" * 60)
print("Testing Images")
print("=" * 60)

test_files = [
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(valid_extensions)
]

if len(test_files) == 0:
    print("ERROR: No test images found.")
    print(f"Put test images inside: {TEST_DIR}/")
    exit()


known_count = 0
unknown_count = 0
no_face_count = 0


for filename in sorted(test_files):

    image_path = os.path.join(
        TEST_DIR,
        filename
    )

    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot read: {filename}")
        continue

    embedding, face = get_embedding(image)

    if embedding is None:
        print(
            f"{filename:<30} "
            f"NO FACE DETECTED"
        )

        no_face_count += 1
        continue


    # --------------------------------------------------------
    # Compare against every employee
    # --------------------------------------------------------

    best_name = None
    best_score = -1.0

    for name, employee_embedding in zip(
        employee_names,
        employee_embeddings
    ):

        score = cosine_similarity(
            embedding,
            employee_embedding
        )

        if score > best_score:
            best_score = score
            best_name = name


    # --------------------------------------------------------
    # Known / Unknown decision
    # --------------------------------------------------------

    if best_score >= THRESHOLD:

        result = "KNOWN"
        known_count += 1

    else:

        result = "UNKNOWN"
        best_name = "Unknown"
        unknown_count += 1


    print(
        f"{filename:<30} "
        f"{result:<10} "
        f"{best_name:<20} "
        f"similarity: {best_score:.4f}"
    )


# ============================================================
# SUMMARY
# ============================================================

total = known_count + unknown_count + no_face_count

print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total test images : {total}")
print(f"Known             : {known_count}")
print(f"Unknown           : {unknown_count}")
print(f"No face           : {no_face_count}")

print(f"\nThreshold         : {THRESHOLD}")

print("=" * 60)
print("Done.")
print("=" * 60)