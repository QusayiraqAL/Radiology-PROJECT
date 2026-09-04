# -*- coding: utf-8 -*-
"""
Brain v3 — fixes the robustness gap external validation exposed.

External test (eval_external/eval_brain.py) showed v2 scores 98.9% on same-source held-out but
only 65.8% on genuinely-novel augmented/re-encoded images (Roboflow re-exports), with pituitary
recall collapsing to 34% (mostly misread as "no tumor" — the dangerous error). v2's augmentation
(±12° rotation, mild jitter) doesn't cover the rotations / crops / recompression a real upload has.

v3 keeps everything from v2 (leak-free grouped split, crop, dropout, label smoothing, early stop)
and only STRENGTHENS augmentation: wider rotation, random-resized-crop, stronger jitter, occasional
blur, and random erasing. Same split + protocol so the before/after is fair.
"""
import os, io, json, time
import numpy as np
import pyarrow.parquet as pq
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score, classification_report

from nets import build_brain_resnet
from img_utils import crop_brain_region
from brain_split import dhash_bits, cluster_near_duplicates, grouped_split, HAMMING_T

HERE = os.path.dirname(__file__)
PARQUET = os.path.join(HERE, "data", "brain_parquet", "data", "train-00000-of-00001.parquet")
MODEL_DIR = os.path.join(HERE, "models")
SIZE, SEED, EPOCHS, BATCH, LR = 128, 0, 16, 32, 5e-4
PATIENCE, LABEL_SMOOTH, DROPOUT = 5, 0.05, 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# STRONGER augmentation than v2 (the only change that matters for the fix)
train_tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((146, 146)),
    transforms.RandomResizedCrop(SIZE, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IM_MEAN, IM_STD),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.12)),
])
eval_tf = transforms.Compose([
    transforms.Grayscale(3), transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
])


class DS(Dataset):
    def __init__(self, imgs, labels, idx, tf): self.imgs, self.labels, self.idx, self.tf = imgs, labels, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        return self.tf(self.imgs[j]), int(self.labels[j])


def main():
    t0 = time.time()
    print("[data] decoding + hashing + cropping ...", flush=True)
    t = pq.read_table(PARQUET).to_pydict()
    raw, labels = t["image"], np.array(t["label"], dtype=np.int64)
    imgs, bits = [], []
    for cell in raw:
        b = cell["bytes"] if isinstance(cell, dict) else cell
        im = Image.open(io.BytesIO(b)).convert("RGB")
        bits.append(dhash_bits(im)); imgs.append(crop_brain_region(im))
    bits = np.stack(bits)
    cid = cluster_near_duplicates(bits, thresh=HAMMING_T)
    tr, va, te = grouped_split(cid, labels, seed=SEED)
    print(f"[split] train={len(tr)} val={len(va)} test={len(te)} (grouped, leak-free)", flush=True)

    tl = DataLoader(DS(imgs, labels, tr, train_tf), batch_size=BATCH, shuffle=True)
    vl = DataLoader(DS(imgs, labels, va, eval_tf), batch_size=64)
    el = DataLoader(DS(imgs, labels, te, eval_tf), batch_size=64)

    counts = np.bincount([labels[i] for i in tr], minlength=4)
    w = counts.sum() / (4 * np.maximum(counts, 1))
    model = build_brain_resnet(4, pretrained=True, dropout=DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE), label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    @torch.no_grad()
    def collect(loader):
        model.eval(); ys, ps = [], []
        for xb, yb in loader:
            ps.append(torch.softmax(model(xb.to(DEVICE)), 1).cpu().numpy()); ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    best_loss, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); tot += loss.item()*len(xb)
        sched.step()
        yv, pv = collect(vl)
        vloss = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-9, 1)).mean())
        acc = accuracy_score(yv, pv.argmax(1))
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(tr):.4f} val_loss={vloss:.4f} val_acc={acc:.4f}", flush=True)
        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  [early stop] patience {PATIENCE}", flush=True); break
    model.load_state_dict(best_state)

    yt, pt = collect(el); pred = pt.argmax(1)
    try: auc = roc_auc_score(yt, pt, multi_class="ovr", average="macro")
    except Exception: auc = float("nan")
    trl = DataLoader(DS(imgs, labels, tr, eval_tf), batch_size=64)
    ytr, ptr = collect(trl)
    metrics = {
        "model": "brain_tumor_mri_resnet18_v3_strongaug", "modality": "brain MRI",
        "dataset": "Brain Tumor MRI (Hemg) — grouped leak-free split (same as v2)",
        "fix": ("v3 strengthens augmentation (rotation 25, random-resized-crop, jitter 0.3, blur, "
                "random-erasing) to close the robustness gap external validation found on augmented images."),
        "anti_overfitting": ["transfer learning", "brain-region cropping", "STRONG augmentation (v3)",
                             f"dropout={DROPOUT}", f"label_smoothing={LABEL_SMOOTH}", "weight_decay=1e-4",
                             "cosine LR", f"early stop patience={PATIENCE}", "class weights"],
        "split": f"GROUPED 70/15/15 by dHash cluster (Hamming<={HAMMING_T}), seed=0 (identical to v2)",
        "input_size": SIZE, "classes": CLASSES,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "test_accuracy": round(accuracy_score(yt, pred), 4),
        "train_accuracy": round(accuracy_score(ytr, ptr.argmax(1)), 4),
        "overfitting_gap": round(accuracy_score(ytr, ptr.argmax(1)) - accuracy_score(yt, pred), 4),
        "test_auc_macro_ovr": None if np.isnan(auc) else round(float(auc), 4),
        "test_macro_f1": round(f1_score(yt, pred, average="macro"), 4),
        "mean_confidence": round(float(pt.max(1).mean()), 4),
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "train_seconds": round(time.time()-t0, 1), "device": DEVICE,
    }
    torch.save({"state_dict": model.state_dict(), "size": SIZE, "classes": CLASSES,
                "cropped": True, "dropout": DROPOUT}, os.path.join(MODEL_DIR, "brain_tumor_mri_v3.pt"))
    json.dump(metrics, open(os.path.join(MODEL_DIR, "brain_v3_metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[classification_report]\n", classification_report(yt, pred, target_names=CLASSES))
    print("[RESULT] v3 test_acc=%.4f gap=%+.4f macroAUC=%s" % (
        metrics["test_accuracy"], metrics["overfitting_gap"], metrics["test_auc_macro_ovr"]))
    print("BRAIN_V3_DONE", flush=True)


if __name__ == "__main__":
    main()
