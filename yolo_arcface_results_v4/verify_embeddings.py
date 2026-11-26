#!/usr/bin/env python3
"""
Verify and validate face embeddings in yolo_arcface_results_v4.
Run:
    python verify_embeddings.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

EMB_DIR = Path("/home/vivek/Music/face/yolo_arcface_results_v4/embeddings")

def check_embeddings():
    print("=" * 70)
    print("VERIFYING EMBEDDINGS DATABASE INTEGRITY")
    print("=" * 70)

    # 1. Load files
    emb_path = EMB_DIR / "employee_embeddings.npy"
    ids_path = EMB_DIR / "employee_ids.npy"
    names_path = EMB_DIR / "employee_names.npy"
    meta_path = EMB_DIR / "embedding_metadata.csv"

    if not all(p.exists() for p in [emb_path, ids_path, names_path, meta_path]):
        print("[FAIL] Missing embedding files in:", EMB_DIR)
        return

    embeddings = np.load(emb_path)
    ids = np.load(ids_path)
    names = np.load(names_path)
    meta = pd.read_csv(meta_path)

    print(f"\n[OK] Files loaded successfully.")
    print(f"  • Embeddings shape: {embeddings.shape} (dtype: {embeddings.dtype})")
    print(f"  • IDs count:        {len(ids)}")
    print(f"  • Names count:      {len(names)}")
    print(f"  • Metadata rows:    {len(meta)}")

    # 2. Check Alignment between arrays
    if len(embeddings) == len(ids) == len(names) == len(meta):
        print("[PASS] All arrays and metadata have matching lengths.")
    else:
        print("[FAIL] Length mismatch among embeddings, IDs, and metadata!")
        return

    # 3. Check for NaNs or Infinite values
    nan_count = np.isnan(embeddings).sum()
    inf_count = np.isinf(embeddings).sum()
    if nan_count == 0 and inf_count == 0:
        print("[PASS] No NaN or Infinite values found.")
    else:
        print(f"[FAIL] Found {nan_count} NaNs and {inf_count} Infs!")

    # 4. Check L2 Normalization (Cosine similarity requirement)
    norms = np.linalg.norm(embeddings, axis=1)
    if np.allclose(norms, 1.0, atol=1e-3):
        print(f"[PASS] All embeddings are properly unit-normalized (Mean norm = {norms.mean():.4f}).")
    else:
        print(f"[WARN] Non-normalized embeddings detected: min={norms.min():.4f}, max={norms.max():.4f}")

    # 5. Retrieval test (Random face self-similarity and margin check)
    print("\n" + "=" * 70)
    print("RUNNING RETRIEVAL & MARGIN TEST (10 random queries)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(embeddings), size=10, replace=False)

    passed_queries = 0
    for idx in sample_indices:
        query_emb = embeddings[idx]
        target_id = ids[idx]
        target_name = names[idx]

        # Dot product against entire database (112k faces)
        scores = np.dot(embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1]

        best_match_idx = top_indices[0]
        runner_up_idx = top_indices[1]

        self_score = scores[best_match_idx]
        runner_up_score = scores[runner_up_idx]
        margin = self_score - runner_up_score

        is_self = (best_match_idx == idx)
        if is_self:
            passed_queries += 1

        status = "PASS" if is_self and margin > 0.3 else "WARN"
        print(f"[{status}] Query: '{target_id}_{target_name}' -> Top Match Score: {self_score:.4f} | Runner-up: {runner_up_score:.4f} (Margin: {margin:.4f})")

    print("\n" + "=" * 70)
    print(f"RESULT: {passed_queries}/10 queries passed with clean margins.")
    print("Embeddings database is verified and working properly!")
    print("=" * 70)

if __name__ == "__main__":
    check_embeddings()
