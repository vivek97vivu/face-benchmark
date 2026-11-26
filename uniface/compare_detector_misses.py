import pandas as pd
from pathlib import Path

RESULT_DIR = Path(
    "/home/vivek/Music/face/insightface_benchmark_results"
)

df = pd.read_csv(RESULT_DIR / "insightface_per_image.csv")

models = [
    "antelopev2",
    "buffalo_l",
    "buffalo_m",
    "buffalo_s"
]

missed = {}

for model in models:

    model_df = df[df["model"] == model]

    missed[model] = set(
        model_df.loc[
            model_df["status"] == "NO_FACE_DETECTED",
            "image"
        ]
    )

    print(f"{model}: {len(missed[model])} missed images")

print("\nPAIRWISE MISS COMPARISON")
print("=" * 65)

for model in models:

    print(f"\n{model}")

    for other in models:

        if model == other:
            continue

        common = missed[model] & missed[other]

        print(
            f"  Same missed as {other}: "
            f"{len(common)}"
        )

all_missed = set.union(*missed.values())

print("\nUNIQUE MISSED IMAGES:", len(all_missed))

print("\nImages missed by all four models:")

common_all = set.intersection(*missed.values())

for image in sorted(common_all):
    print(image)

print("\nImages missed by at least one model:")

for image in sorted(all_missed):
    print(image)

output = RESULT_DIR / "detector_miss_comparison.txt"

with open(output, "w") as f:

    f.write(f"Unique missed images: {len(all_missed)}\n")
    f.write(f"Missed by all four: {len(common_all)}\n\n")

    for image in sorted(all_missed):
        missed_by = [
            model for model in models
            if image in missed[model]
        ]

        f.write(f"{image}: {', '.join(missed_by)}\n")

print("\nSaved:", output)
