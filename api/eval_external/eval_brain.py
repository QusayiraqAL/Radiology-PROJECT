# -*- coding: utf-8 -*-
"""EXTERNAL validation for the brain-MRI model.

Reality (measured, not assumed): every public 4-class brain-tumor-MRI set traces to the SAME
~7023-image pool (Simezu is byte-for-class identical; PranomVignesh is that pool re-exported by
Roboflow). So a truly independent 2500-image cohort does not exist publicly for this task.

What we do instead, honestly:
  1. Reference = the images this model actually TRAINED on (Hemg grouped-train).
  2. Candidates = (a) our leak-free grouped HELD-OUT (699, never trained on) +
                  (b) PranomVignesh images that are dHash-NOVEL vs the training set.
  3. dHash-dedup candidates vs the training reference (Hamming > 5 = genuinely unseen).
  4. Run the SAME served model (crop + resize + ImageNet norm) and report the confusion matrix.

The notebook states the composition of the test set explicitly.
"""
import os, sys, io, time
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_brain, hash_matrix, min_hamming_to_ref, save_results, API
from brain_split import cluster_near_duplicates, grouped_split, HAMMING_T

HEMG = os.path.join(API, "data", "brain_parquet", "data", "train-00000-of-00001.parquet")
PRANO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_dl", "prano", "timri")
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
SEED, TARGET = 0, 2500
BRAIN_CKPT = os.environ.get("BRAIN_CKPT", "brain_tumor_mri_v2.pt")
OUT_NAME = os.environ.get("BRAIN_OUT", "brain")

# map various folder spellings -> our canonical class name
NAME2CLS = {"glioma": "glioma", "meningioma": "meningioma", "pituitary": "pituitary",
            "notumor": "notumor", "no_tumor": "notumor", "no-tumor": "notumor",
            "1-notumor": "notumor", "2-glioma": "glioma", "3-meningioma": "meningioma",
            "4-pituitary": "pituitary"}


def load_hemg():
    t = pq.read_table(HEMG).to_pydict()
    imgs, labels = [], np.array(t["label"], np.int64)
    for cell in t["image"]:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        imgs.append(Image.open(io.BytesIO(b)).convert("RGB"))
    return imgs, labels


def load_prano():
    imgs, labels = [], []
    if not os.path.isdir(PRANO):
        print("[brain] PranomVignesh not downloaded; using held-out only", flush=True)
        return imgs, labels
    for split in os.listdir(PRANO):
        sp = os.path.join(PRANO, split)
        if not os.path.isdir(sp):
            continue
        for cls_dir in os.listdir(sp):
            cls = NAME2CLS.get(cls_dir.lower())
            if cls is None:
                continue
            d = os.path.join(sp, cls_dir)
            for fn in os.listdir(d):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        imgs.append(Image.open(os.path.join(d, fn)).convert("RGB"))
                        labels.append(CLASSES.index(cls))
                    except Exception:
                        pass
    return imgs, np.array(labels, np.int64) if labels else np.array([], np.int64)


def main():
    t0 = time.time()
    predict, model_classes = load_brain(BRAIN_CKPT)
    assert model_classes == CLASSES, f"class order mismatch: {model_classes}"
    print(f"[brain] checkpoint={BRAIN_CKPT} -> results/{OUT_NAME}", flush=True)

    print("[brain] loading + hashing Hemg (training source) ...", flush=True)
    hemg_imgs, hemg_lab = load_hemg()
    hemg_bits = hash_matrix(hemg_imgs)
    # reproduce the EXACT training split so we dedup vs what the model actually saw
    cid = cluster_near_duplicates(hemg_bits, thresh=HAMMING_T)
    tr, va, te = grouped_split(cid, hemg_lab, seed=SEED)
    train_bits = hemg_bits[tr]
    print(f"[brain] hemg={len(hemg_imgs)} train_ref={len(tr)} heldout(te)={len(te)}", flush=True)

    # candidate set (a): our leak-free held-out — genuinely unseen by construction
    cand_imgs = [hemg_imgs[i] for i in te]
    cand_lab = [int(hemg_lab[i]) for i in te]
    cand_src = ["heldout"] * len(te)

    # candidate set (b): PranomVignesh images that are dHash-NOVEL vs training
    p_imgs, p_lab = load_prano()
    if len(p_imgs):
        p_bits = hash_matrix(p_imgs)
        d = min_hamming_to_ref(p_bits, train_bits)
        novel = np.where(d > HAMMING_T)[0]
        print(f"[brain] PranomVignesh={len(p_imgs)} novel-vs-train(>{HAMMING_T})={len(novel)} "
              f"({100*len(novel)/len(p_imgs):.1f}%)", flush=True)
        for i in novel:
            cand_imgs.append(p_imgs[i]); cand_lab.append(int(p_lab[i])); cand_src.append("prano_novel")

    cand_lab = np.array(cand_lab)
    # cap to TARGET (keep all held-out, fill with prano-novel), stratified-ish by source order
    if len(cand_imgs) > TARGET:
        keep = list(range(len(cand_imgs)))
        rng = np.random.RandomState(SEED); rng.shuffle(keep)
        # prioritize held-out, then prano
        keep = sorted(keep, key=lambda i: 0 if cand_src[i] == "heldout" else 1)[:TARGET]
        cand_imgs = [cand_imgs[i] for i in keep]; cand_lab = cand_lab[keep]
        cand_src = [cand_src[i] for i in keep]

    from collections import Counter
    comp = Counter(cand_src)
    print(f"[brain] test set = {len(cand_imgs)} images | composition={dict(comp)}", flush=True)

    prob = predict(cand_imgs)
    pred = prob.argmax(1)
    acc = float((pred == cand_lab).mean())
    src_arr = np.array(cand_src)
    # per-source breakdown — the blended number hides two very different populations
    by_source = {}
    for s in sorted(set(cand_src)):
        m = src_arr == s
        by_source[s] = {"n": int(m.sum()), "accuracy": round(float((pred[m] == cand_lab[m]).mean()), 4)}
    print(f"[brain] accuracy={acc:.4f} mean_conf={prob.max(1).mean():.4f} | by_source={by_source} "
          f"in {time.time()-t0:.0f}s", flush=True)

    save_results(OUT_NAME, {
        "model": f"brain_tumor_mri ({BRAIN_CKPT})",
        "modality": "brain MRI", "task": "4-class tumor classification",
        "dataset": "leak-free held-out (Hemg) + PranomVignesh novel-vs-train (dHash>5)",
        "is_external_source": True,
        "monoculture_note": ("All public 4-class brain-MRI sets are the same ~7023-image pool "
                             "(Simezu identical; PranomVignesh = Roboflow re-export). A fully "
                             "independent cohort does not exist publicly; composition is reported."),
        "test_composition": dict(comp),
        "accuracy_by_source": by_source,
        "classes": CLASSES, "class_names_ar": ["ورم دبقي", "ورم سحائي", "لا ورم", "ورم نخامية"],
        "n_test": int(len(cand_imgs)), "accuracy": round(acc, 4),
        # Guarded the same way test_auc above is. An fp16 eval pass can hand back
        # non-finite probabilities, and json.dump writes a bare NaN token that json.load
        # reads back without complaint - so the bad value survives into the API, where
        # Starlette serializes with allow_nan=False and kills the whole /models response.
        # derma_v2 and derma_bin both shipped one for two days before it surfaced.
        "mean_confidence": (round(float(prob.max(1).mean()), 4)
                            if np.isfinite(prob).all() else None),
    }, y_true=cand_lab, y_pred=pred, y_prob=prob, extra_arrays={"source": src_arr})


if __name__ == "__main__":
    main()
