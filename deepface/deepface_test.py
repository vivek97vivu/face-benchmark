import os

# ============================================================
# FORCE TENSORFLOW TO CPU
# Must be set BEFORE importing TensorFlow / DeepFace
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import time
from deepface import DeepFace


# ============================================================
# CONFIG
# ============================================================

IMG = "/home/vivek/Music/face/employee/vivek.jpg"


# ============================================================
# TEST
# ============================================================

print("=" * 60)
print("GhostFaceNet CPU Test")
print("=" * 60)

print("Image:", IMG)
print("Device: CPU")


start = time.perf_counter()

result = DeepFace.represent(
    img_path=IMG,
    model_name="GhostFaceNet",
    detector_backend="opencv",
    enforce_detection=True,
    align=True,
)

elapsed = time.perf_counter() - start


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print("Faces detected :", len(result))
print("Total time     :", f"{elapsed:.3f} sec")

if result:

    embedding = result[0]["embedding"]

    print("Embedding size :", len(embedding))
    print("Embedding type :", type(embedding))

    print()
    print("First 10 embedding values:")

    for i, value in enumerate(embedding[:10]):
        print(f"{i:2d}: {value:.8f}")

print("=" * 60)
