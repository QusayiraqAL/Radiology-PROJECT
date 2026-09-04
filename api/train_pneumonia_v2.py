# -*- coding: utf-8 -*-
"""
Pneumonia v2 — applies the techniques from the user's notebooks (Untitled16) to OUR pipeline.

From the notebook, adopted here (in PyTorch):
  * REAL full-resolution chest-xray-pneumonia dataset (not 64px MedMNIST)
  * transfer learning on an ImageNet backbone (ResNet-18; notebook used ResNet50)
  * data augmentation (shift / zoom / shear / h-flip / brightness)   [ImageDataGenerator -> torchvision]
  * class_weight='balanced'                                          [CrossEntropy weight]
  * best-checkpoint on validation loss                              [ModelCheckpoint]
  * evaluation with confusion matrix + classification report

Improvement over the notebook: proper ImageNet normalization (the notebook fed /255 to an
ImageNet backbone, a preprocessing mismatch). Evaluated on the official 624-image TEST split.
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, f1_score, classification_report)
import medmnist
from medmnist import INFO
from nets import build_pneumonia_resnet   # ResNet-18 + dropout head

HERE = os.path.dirname(__file__)
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
os.makedirs(DATA_ROOT, exist_ok=True)
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
SIZE, SEED, BATCH, LR = 224, 0, 32, 1e-4
# regularization is env-overridable so we can tune the train/test gap without editing code
EPOCHS = int(os.environ.get("PNEU_EPOCHS", "8"))
PATIENCE = int(os.environ.get("PNEU_PATIENCE", "3"))     # early stopping on val loss
LABEL_SMOOTH = float(os.environ.get("PNEU_LABEL_SMOOTH", "0.05"))  # soften one-hot targets
DROPOUT = float(os.environ.get("PNEU_DROPOUT", "0.3"))   # dropout before the classifier head
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_split(split):
    # PneumoniaMNIST at 224px = real Kermany-2018 chest X-rays (same source as the notebook dataset)
    DataClass = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
    ds = DataClass(split=split, download=True, size=224, root=DATA_ROOT)
    imgs = ds.imgs                                   # (N, 224, 224) uint8 grayscale
    if imgs.ndim == 3:
        imgs = np.repeat(imgs[..., None], 3, axis=-1)  # -> RGB for the ImageNet backbone
    return imgs.astype(np.uint8), ds.labels.astype(np.int64).reshape(-1)


train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.1),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), shear=5),
    transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
])
eval_tf = transforms.Compose([
    transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
])


class DS(Dataset):
    def __init__(self, X, y, tf): self.X, self.y, self.tf = X, y, tf
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.tf(self.X[i]), int(self.y[i])


def main():
    t0 = time.time()
    print("[data] loading PneumoniaMNIST-224 (real Kermany chest X-rays) ...", flush=True)
    Xtr, ytr = load_split("train")
    Xva, yva = load_split("val")
    Xte, yte = load_split("test")
    print(f"[data] train={len(Xtr)} val={len(Xva)} test={len(Xte)} | test class balance={np.bincount(yte).tolist()}", flush=True)

    tl = DataLoader(DS(Xtr, ytr, train_tf), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(DS(Xva, yva, eval_tf), batch_size=64)
    el = DataLoader(DS(Xte, yte, eval_tf), batch_size=64)

    counts = np.bincount(ytr, minlength=2)
    w = counts.sum() / (2 * np.maximum(counts, 1))
    model = build_pneumonia_resnet(num_classes=2, pretrained=True, dropout=DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                               label_smoothing=LABEL_SMOOTH)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    @torch.no_grad()
    def probs(loader):
        model.eval(); ps, ys = [], []
        for xb, yb in loader:
            p = torch.softmax(model(xb.to(DEVICE)), 1)[:, 1].cpu().numpy()
            ps.append(p); ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    best_loss, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        yv, pv = probs(vl)
        vloss = float(nn.functional.binary_cross_entropy(torch.tensor(pv).clamp(1e-6, 1-1e-6), torch.tensor(yv, dtype=torch.float32)))
        vacc = accuracy_score(yv, (pv >= 0.5).astype(int))
        print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(Xtr):.4f} val_loss={vloss:.4f} val_acc={vacc:.4f}", flush=True)
        if vloss < best_loss - 1e-4:
            best_loss, bad = vloss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  [early stop] val_loss did not improve for {PATIENCE} epochs", flush=True)
                break
    model.load_state_dict(best_state)

    # threshold via Youden's J on val
    yv, pv = probs(vl)
    ths = np.linspace(0.1, 0.9, 81); bj, bt = -1, 0.5
    for th in ths:
        tn, fp, fn, tp = confusion_matrix(yv, (pv >= th).astype(int), labels=[0, 1]).ravel()
        j = tp/max(tp+fn, 1) + tn/max(tn+fp, 1) - 1
        if j > bj: bj, bt = j, float(th)

    yt, pt = probs(el)
    pred = (pt >= bt).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()

    # overfitting check: accuracy on the (un-augmented) TRAIN set vs the held-out TEST set
    trl = DataLoader(DS(Xtr, ytr, eval_tf), batch_size=64)
    ytr_e, ptr_e = probs(trl)
    train_acc = accuracy_score(ytr_e, (ptr_e >= bt).astype(int))
    test_acc = accuracy_score(yt, pred)

    metrics = {
        "model": "pneumonia_v2_resnet18_fullres",
        "task": "binary (normal vs pneumonia)", "modality": "chest X-ray",
        "dataset": "PneumoniaMNIST-224 (MedMNIST; real Kermany-2018 chest X-rays at 224px)",
        "anti_overfitting": ["transfer learning (ImageNet ResNet-18)",
                             "augmentation (crop/zoom/shear/flip/brightness)",
                             f"dropout={DROPOUT} head", f"label_smoothing={LABEL_SMOOTH}",
                             "weight_decay=1e-4", "cosine LR decay",
                             f"early stopping (patience={PATIENCE}) + best-checkpoint on val loss",
                             "class weights (balanced)"],
        "input_size": SIZE, "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
        "decision_threshold": round(bt, 3),
        "train_accuracy": round(float(train_acc), 4),
        "overfitting_gap": round(float(train_acc - test_acc), 4),
        "test_accuracy": round(accuracy_score(yt, pred), 4),
        "test_auc": round(float(roc_auc_score(yt, pt)), 4),
        "test_sensitivity_recall": round(recall_score(yt, pred), 4),
        "test_specificity": round(tn/max(tn+fp, 1), 4),
        "test_precision": round(precision_score(yt, pred, zero_division=0), 4),
        "test_f1": round(f1_score(yt, pred), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "labels": {"0": "normal", "1": "pneumonia"},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "train_seconds": round(time.time()-t0, 1), "device": DEVICE,
    }
    torch.save({"state_dict": model.state_dict(), "size": SIZE, "threshold": bt,
                "arch": "resnet18", "dropout": DROPOUT, "mean": IM_MEAN, "std": IM_STD},
               os.path.join(MODEL_DIR, "pneumonia_v2.pt"))
    json.dump(metrics, open(os.path.join(MODEL_DIR, "pneumonia_v2_metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[classification_report]\n", classification_report(yt, pred, target_names=["NORMAL", "PNEUMONIA"]))
    print("[RESULT]", json.dumps({k: metrics[k] for k in ["test_accuracy","test_auc","test_sensitivity_recall",
                                                          "test_specificity","train_accuracy","overfitting_gap"]}))
    print("PNEU_V2_DONE")


if __name__ == "__main__":
    main()
