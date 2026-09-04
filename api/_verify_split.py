# -*- coding: utf-8 -*-
"""Prove the grouped split removes the leakage the per-image split had.
Re-runs the SAME dHash audit on both splits and prints them side by side."""
import os, io, json, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from brain_split import dhash_bits, cluster_near_duplicates, grouped_split, pairwise_min_dist

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "data", "brain_parquet", "data", "train-00000-of-00001.parquet")
SEED = 0


def per_image_split(labels, seed=SEED):
    """The ORIGINAL (v1/v2) split logic — audited here for comparison."""
    rng = np.random.RandomState(seed); tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]; rng.shuffle(idx)
        n = len(idx); nte = int(0.15 * n); nva = int(0.15 * n)
        te += idx[:nte].tolist(); va += idx[nte:nte+nva].tolist(); tr += idx[nte+nva:].tolist()
    return tr, va, te


def audit(name, tr, te, bits, labels):
    d = pairwise_min_dist(bits[te], bits[tr])
    print(f"[{name}]  n_train={len(tr):5} n_test={len(te):5}  "
          f"exact_dup={np.mean(d == 0)*100:5.1f}%  near<=3={np.mean(d <= 3)*100:5.1f}%  "
          f"near<=5={np.mean(d <= 5)*100:5.1f}%  mean_min_dist={d.mean():.2f}", flush=True)
    print(f"          test class balance={np.bincount(labels[te], minlength=4).tolist()}", flush=True)
    return {"n_train": len(tr), "n_test": len(te),
            "exact_dup_rate": round(float(np.mean(d == 0)), 4),
            "near3_rate": round(float(np.mean(d <= 3)), 4),
            "near5_rate": round(float(np.mean(d <= 5)), 4),
            "mean_min_hamming": round(float(d.mean()), 2)}


def main():
    t0 = time.time()
    t = pq.read_table(PARQUET).to_pydict()
    raw, labels = t["image"], np.array(t["label"], dtype=np.int64)
    bits = np.stack([dhash_bits(Image.open(io.BytesIO(
        c["bytes"] if isinstance(c, dict) else c))) for c in raw])
    print(f"[data] n={len(bits)} hashed in {time.time()-t0:.0f}s", flush=True)

    tr1, va1, te1 = per_image_split(labels)
    r_old = audit("per-image  (v1/v2 current)", tr1, te1, bits, labels)

    t1 = time.time()
    cid = cluster_near_duplicates(bits)
    n_cl = cid.max() + 1
    sizes = np.bincount(cid)
    print(f"[cluster] {len(bits)} images -> {n_cl} clusters in {time.time()-t1:.0f}s "
          f"(largest={sizes.max()}, singletons={(sizes == 1).sum()})", flush=True)

    tr2, va2, te2 = grouped_split(cid, labels, seed=SEED)
    r_new = audit("grouped    (dedup by cluster)", tr2, te2, bits, labels)

    json.dump({"per_image_split": r_old, "grouped_split": r_new,
               "n_clusters": int(n_cl), "n_images": int(len(bits)),
               "largest_cluster": int(sizes.max()),
               "singleton_clusters": int((sizes == 1).sum())},
              open(os.path.join(HERE, "models", "split_comparison.json"), "w"), indent=2)
    print("VERIFY_SPLIT_DONE", flush=True)


if __name__ == "__main__":
    main()
