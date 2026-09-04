# -*- coding: utf-8 -*-
"""
Is the brain model's 98.95% real, or inflated by leakage?

This public dataset (Hemg/Brain-Tumor-MRI-Dataset) ships NO patient IDs, so the
70/15/15 split is per-IMAGE. Consecutive MRI slices of the SAME patient are nearly
identical — if some land in train and others in test, the test set is not truly
"unseen" and accuracy is optimistically biased.

We cannot recover patient IDs, but we CAN measure the near-duplicate rate across the
split boundary with perceptual hashing (dHash). That bounds how much of the score
could be explained by leakage.

Output: models/brain_leakage_report.json
"""
import os, io, json, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "data", "brain_parquet", "data", "train-00000-of-00001.parquet")
OUT = os.path.join(HERE, "models", "brain_leakage_report.json")
SEED = 0
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def dhash_bits(pil, size=8):
    """Difference hash -> 64 bits as a uint8 vector (kept as bits for vectorized Hamming)."""
    g = pil.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


def stratified_split(labels, seed=SEED):
    """IDENTICAL split logic to train_brain_v2.py — we must audit the real split."""
    rng = np.random.RandomState(seed); tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        n = len(idx); nte = int(0.15 * n); nva = int(0.15 * n)
        te += idx[:nte].tolist(); va += idx[nte:nte+nva].tolist(); tr += idx[nte+nva:].tolist()
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def main():
    t0 = time.time()
    print("[data] decoding + hashing brain images ...", flush=True)
    t = pq.read_table(PARQUET).to_pydict()
    raw, labels = t["image"], np.array(t["label"], dtype=np.int64)
    bits = np.stack([dhash_bits(Image.open(io.BytesIO(
        cell["bytes"] if isinstance(cell, dict) else cell))) for cell in raw])
    n = len(bits)
    print(f"[data] n={n} hashed in {time.time()-t0:.0f}s", flush=True)

    tr, va, te = stratified_split(labels)
    B_tr = bits[tr].astype(np.int16)      # (Ntr, 64)
    B_te = bits[te].astype(np.int16)      # (Nte, 64)

    # Hamming distance via matrix algebra: d = popcount(a XOR b)
    #   = sum(a) + sum(b) - 2*<a,b>   for 0/1 vectors
    sa = B_te.sum(1, keepdims=True)                    # (Nte,1)
    sb = B_tr.sum(1, keepdims=True).T                  # (1,Ntr)
    dists = np.empty(len(te), dtype=np.int16)
    CH = 256
    for i in range(0, len(te), CH):
        blk = B_te[i:i+CH]
        d = sa[i:i+CH] + sb - 2 * (blk @ B_tr.T)       # (chunk, Ntr)
        dists[i:i+CH] = d.min(1)

    exact = int((dists == 0).sum())
    near3 = int((dists <= 3).sum())
    near5 = int((dists <= 5).sum())
    report = {
        "question": "Is the held-out brain test set contaminated by near-duplicate slices from train?",
        "method": "dHash (64-bit perceptual hash), Hamming distance from each TEST image to nearest TRAIN image",
        "split": "per-image stratified 70/15/15, seed=0 (identical to train_brain_v2.py)",
        "n_total": int(n), "n_train": len(tr), "n_test": len(te),
        "exact_duplicate_hash_in_train": exact,
        "exact_duplicate_rate": round(exact / len(te), 4),
        "near_duplicate_le3_count": near3,
        "near_duplicate_le3_rate": round(near3 / len(te), 4),
        "near_duplicate_le5_count": near5,
        "near_duplicate_le5_rate": round(near5 / len(te), 4),
        "min_hamming_mean": round(float(dists.mean()), 2),
        "min_hamming_median": float(np.median(dists)),
        "caveat": ("Dataset ships no patient IDs, so a per-patient split is impossible. "
                   "A high near-duplicate rate means the reported accuracy is an optimistic "
                   "upper bound rather than a patient-level generalization estimate."),
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in [
        "exact_duplicate_rate", "near_duplicate_le3_rate", "near_duplicate_le5_rate",
        "min_hamming_mean", "min_hamming_median"]}, indent=2))
    print("LEAKAGE_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
