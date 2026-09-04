# -*- coding: utf-8 -*-
"""
Honest splitting for the brain-MRI dataset.

WHY: check_brain_leakage.py measured that 22% of the per-image test split has an EXACT
perceptual-hash twin in train, and 66% has a near-twin (Hamming<=5). Consecutive MRI
slices of one patient are near-identical, and this public dataset ships no patient IDs,
so a per-image split silently puts the same patient on both sides. The resulting 98.95%
is an optimistic upper bound, not generalization.

FIX: group near-duplicate images into clusters (union-find over dHash Hamming<=T), then
split CLUSTERS — never splitting a cluster across train/test. Clusters approximate
"same patient / same acquisition", so this is a patient-level split in spirit.

Shared by check_brain_leakage.py and train_brain_v2.py so the audit and the training
use exactly the same grouping.
"""
import numpy as np
from PIL import Image

HAMMING_T = 5          # <= this distance counts as "same source slice/patient"


def dhash_bits(pil, size=8):
    """Difference hash -> 64 bits as uint8 vector."""
    g = pil.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


def pairwise_min_dist(B_query, B_ref, chunk=256):
    """Min Hamming distance from each row of B_query to any row of B_ref.
    Uses d = sum(a) + sum(b) - 2<a,b>, exact for 0/1 vectors."""
    A = B_query.astype(np.int16); R = B_ref.astype(np.int16)
    sa = A.sum(1, keepdims=True); sb = R.sum(1, keepdims=True).T
    out = np.empty(len(A), dtype=np.int16)
    for i in range(0, len(A), chunk):
        d = sa[i:i+chunk] + sb - 2 * (A[i:i+chunk] @ R.T)
        out[i:i+chunk] = d.min(1)
    return out


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_near_duplicates(bits, thresh=HAMMING_T, chunk=512):
    """Union-find over all pairs with Hamming distance <= thresh. Returns cluster id per image."""
    n = len(bits)
    B = bits.astype(np.int16)
    s = B.sum(1, keepdims=True)
    dsu = _DSU(n)
    for i in range(0, n, chunk):
        blk = B[i:i+chunk]
        d = s[i:i+chunk] + s.T - 2 * (blk @ B.T)      # (chunk, n)
        # only look forward to avoid doing every pair twice
        for r in range(d.shape[0]):
            gi = i + r
            js = np.nonzero(d[r] <= thresh)[0]
            for j in js:
                if j > gi:
                    dsu.union(gi, int(j))
    roots = np.array([dsu.find(i) for i in range(n)])
    # compact ids
    _, cid = np.unique(roots, return_inverse=True)
    return cid


def grouped_split(cluster_ids, labels, seed=0, val_frac=0.15, test_frac=0.15):
    """Stratified split over CLUSTERS (a cluster never spans two splits).

    Each cluster is assigned its majority class, then clusters are dealt per class so the
    class balance is preserved while duplicates stay on one side of the boundary.
    """
    rng = np.random.RandomState(seed)
    n_clusters = cluster_ids.max() + 1
    # majority class per cluster
    cl_label = np.zeros(n_clusters, dtype=np.int64)
    for c in range(n_clusters):
        members = labels[cluster_ids == c]
        cl_label[c] = np.bincount(members).argmax()

    tr_c, va_c, te_c = [], [], []
    for cls in np.unique(labels):
        cs = np.where(cl_label == cls)[0]
        rng.shuffle(cs)
        n = len(cs)
        nte = int(round(test_frac * n)); nva = int(round(val_frac * n))
        te_c += cs[:nte].tolist()
        va_c += cs[nte:nte+nva].tolist()
        tr_c += cs[nte+nva:].tolist()

    tr_c, va_c, te_c = set(tr_c), set(va_c), set(te_c)
    tr = [i for i in range(len(labels)) if cluster_ids[i] in tr_c]
    va = [i for i in range(len(labels)) if cluster_ids[i] in va_c]
    te = [i for i in range(len(labels)) if cluster_ids[i] in te_c]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te
