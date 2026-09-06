# -*- coding: utf-8 -*-
"""
Generic MedMNIST trainer — adds new REAL diagnostic models with the same anti-overfitting
recipe as the image models here (transfer learning, augmentation, dropout, label smoothing,
weight decay, cosine LR, early stopping on val loss, class weights) and reports the train/test
gap honestly.

Usage:  DATASET=bloodmnist python train_medmnist.py
        DATASET=dermamnist SIZE=64 python train_medmnist.py
        DATASET=breastmnist python train_medmnist.py

Writes models/<key>.pt  (+ models/<key>_metrics.json) where <key> is the DATASET without the
'mnist' suffix (bloodmnist -> blood). main.py auto-discovers these and serves them.
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

from nets import build_brain_resnet   # ResNet-18 + optional dropout head (num_classes configurable)

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASET = os.environ.get("DATASET", "bloodmnist")
SIZE = int(os.environ.get("SIZE", "64"))
MAX_TRAIN = int(os.environ.get("MAX_TRAIN", "0"))   # >0 = subsample train (CPU speed on big sets)
EPOCHS = int(os.environ.get("EPOCHS", "15"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "5e-4"))
PATIENCE = int(os.environ.get("PATIENCE", "4"))
DROPOUT = float(os.environ.get("DROPOUT", "0.3"))
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
KEY = DATASET.replace("mnist", "")


def load_split(split):
    DataClass = getattr(medmnist, INFO[DATASET]["python_class"])
    ds = DataClass(split=split, download=True, size=SIZE, root=DATA_ROOT)
    imgs = ds.imgs
    if imgs.ndim == 3:                                    # grayscale -> RGB
        imgs = np.repeat(imgs[..., None], 3, axis=-1)
    return imgs.astype(np.uint8), ds.labels.astype(np.int64).reshape(-1)


train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
])
eval_tf = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
])


class DS(Dataset):
    def __init__(self, X, y, tf): self.X, self.y, self.tf = X, y, tf
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.tf(self.X[i]), int(self.y[i])


def main():
    t0 = time.time()
    classes = [INFO[DATASET]["label"][str(i)] for i in range(len(INFO[DATASET]["label"]))]
    n_cls = len(classes)
    print(f"[{KEY}] {DATASET} size={SIZE} classes={n_cls}", flush=True)
    Xtr, ytr = load_split("train"); Xva, yva = load_split("val"); Xte, yte = load_split("test")
    if MAX_TRAIN and len(Xtr) > MAX_TRAIN:              # stratified subsample for CPU speed
        rng = np.random.RandomState(SEED); keep = []
        per = MAX_TRAIN // n_cls
        for c in range(n_cls):
            idx = np.where(ytr == c)[0]
            keep += rng.choice(idx, min(len(idx), per), replace=False).tolist()
        rng.shuffle(keep); Xtr, ytr = Xtr[keep], ytr[keep]
        print(f"[{KEY}] subsampled train -> {len(Xtr)} (MAX_TRAIN={MAX_TRAIN})", flush=True)
    print(f"[{KEY}] train={len(Xtr)} val={len(Xva)} test={len(Xte)} "
          f"| test balance={np.bincount(yte, minlength=n_cls).tolist()}", flush=True)

    tl = DataLoader(DS(Xtr, ytr, train_tf), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(DS(Xva, yva, eval_tf), batch_size=128)
    el = DataLoader(DS(Xte, yte, eval_tf), batch_size=128)
    trl = DataLoader(DS(Xtr, ytr, eval_tf), batch_size=128)   # for train-acc (no aug)

    counts = np.bincount(ytr, minlength=n_cls)
    w = counts.sum() / (n_cls * np.maximum(counts, 1))
    model = build_brain_resnet(num_classes=n_cls, pretrained=True, dropout=DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                               label_smoothing=LABEL_SMOOTH)
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
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(Xtr):.4f} val_loss={vloss:.4f} val_acc={acc:.4f}", flush=True)
        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  [early stop] patience {PATIENCE}", flush=True); break
    model.load_state_dict(best_state)

    yt, pt = collect(el); pred = pt.argmax(1)
    ytr_e, ptr_e = collect(trl)
    train_acc = accuracy_score(ytr_e, ptr_e.argmax(1)); test_acc = accuracy_score(yt, pred)
    try:
        auc = (roc_auc_score(yt, pt[:, 1]) if n_cls == 2
               else roc_auc_score(yt, pt, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")
    metrics = {
        "model": f"{KEY}_resnet18", "dataset_key": KEY, "medmnist": DATASET,
        "modality": INFO[DATASET].get("description", "")[:120],
        "task": f"{n_cls}-class classification",
        "anti_overfitting": ["transfer learning (ImageNet ResNet-18)", "augmentation (crop/flip/jitter/rotate)",
                             f"dropout={DROPOUT}", f"label_smoothing={LABEL_SMOOTH}", "weight_decay=1e-4",
                             "cosine LR", f"early stopping (patience={PATIENCE}) on val loss", "class weights"],
        "input_size": SIZE, "classes": classes, "n_classes": n_cls,
        "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "test_accuracy": round(float(test_acc), 4),
        "train_accuracy": round(float(train_acc), 4),
        "overfitting_gap": round(float(train_acc - test_acc), 4),
        "test_macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
        "test_auc": None if np.isnan(auc) else round(float(auc), 4),
        # Guarded the same way test_auc above is. An fp16 eval pass can hand back
        # non-finite probabilities, and json.dump writes a bare NaN token that json.load
        # reads back without complaint - so the bad value survives into the API, where
        # Starlette serializes with allow_nan=False and kills the whole /models response.
        # derma_v2 and derma_bin both shipped one for two days before it surfaced.
        "mean_confidence": (round(float(pt.max(1).mean()), 4)
                            if np.isfinite(pt).all() else None),
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "train_seconds": round(time.time()-t0, 1), "device": DEVICE,
    }
    torch.save({"state_dict": model.state_dict(), "size": SIZE, "classes": classes,
                "dropout": DROPOUT, "mean": IM_MEAN, "std": IM_STD},
               os.path.join(MODEL_DIR, f"{KEY}.pt"))
    json.dump(metrics, open(os.path.join(MODEL_DIR, f"{KEY}_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[report]\n", classification_report(yt, pred, target_names=classes, zero_division=0))
    print(f"[RESULT] {KEY} test_acc={test_acc:.4f} train_acc={train_acc:.4f} "
          f"gap={train_acc-test_acc:+.4f} macroF1={metrics['test_macro_f1']} auc={metrics['test_auc']}")
    print(f"{KEY.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
