# -*- coding: utf-8 -*-
"""
Ordinal (CORAL) head for RetinaMNIST diabetic-retinopathy grading.

WHY. DR grades 0-4 are ORDERED: grade 0 is closer to 1 than it is to 4. Plain
CrossEntropyLoss does not know that — it treats "predicted 0, truth 4" as exactly the same
mistake as "predicted 0, truth 1". The current retina_v2 confusion matrix says that ordering
information is sitting on the table:

    correct              243/400  (60.8%)
    off by ONE grade     111/400  (27.8%)   <-- the model nearly knows the order already
    off by two or more    46/400  (11.5%)

88.5% of predictions are already correct or one grade away. A loss that is told about the
ordering usually converts a slice of that 27.8% into exact hits.

HOW (CORAL — Cao, Mirjalili & Raschka, 2020). Instead of 5 mutually exclusive logits, learn
K-1 = 4 BINARY questions that share one weight vector and differ only in their bias:

    P(grade > 0), P(grade > 1), P(grade > 2), P(grade > 3)

Sharing the weights is what makes the four answers mutually consistent. Because only the bias
differs between thresholds, the ORDER of the four logits is the same for every input — that is
CORAL's rank-consistency guarantee. Four independent binary heads can contradict themselves
per sample (claiming P(>2) > P(>1) on one image and the reverse on the next); this cannot.
The predicted grade is the count of questions answered yes. Note the guarantee is consistency
of the ordering across inputs, not that the biases are themselves sorted — the loss is what
pushes them into a sensible order, and the script checks whether they ended up sorted.

WHAT IS HELD CONSTANT. Same backbone (ResNet-18 @224), same two-stage schedule, same
augmentation, same optimiser, same seed, same splits as train_medmnist_v2.py. The ONLY change
is the head and the loss, so any difference is attributable to the ordinal formulation rather
than to a better recipe. The script reports accuracy AND the ordinal metrics that actually
matter clinically (off-by-one rate, quadratic-weighted kappa, referable-DR sensitivity).

  python train_retina_ordinal.py
  EPOCHS=40 python train_retina_ordinal.py
"""
import os, json, time, copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report, cohen_kappa_score)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")

DATASET = "retinamnist"
KEY = os.environ.get("KEY", "retina_ord")
SIZE = int(os.environ.get("SIZE", "224"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
WARMUP = int(os.environ.get("WARMUP", "4"))
BATCH = int(os.environ.get("BATCH", "32"))
LR = float(os.environ.get("LR", "3e-4"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
PATIENCE = int(os.environ.get("PATIENCE", "10"))
DROPOUT = float(os.environ.get("DROPOUT", "0.4"))
SEED = int(os.environ.get("SEED", "0"))
AMP = os.environ.get("AMP", "1") == "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = AMP and DEVICE == "cuda"
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = False        # see the note in train_medmnist_v2.py
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
N_GRADES = 5
REFERABLE = 2                                  # grade >= 2 is referable DR

LOG = os.path.join(MODEL_DIR, "_retrain91.log")


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")


class CoralHead(nn.Module):
    """One shared weight vector, K-1 free biases -> monotone cumulative probabilities."""
    def __init__(self, in_features, n_grades, dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc = nn.Linear(in_features, 1, bias=False)      # SHARED weights
        self.bias = nn.Parameter(torch.zeros(n_grades - 1))  # per-threshold bias only

    def forward(self, x):
        return self.fc(self.drop(x)) + self.bias             # (N, K-1) logits


def coral_targets(y, n_grades):
    """y=3 -> [1,1,1,0]: 'is the grade > 0/1/2/3?'"""
    lv = torch.arange(n_grades - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > lv).float()


def coral_loss(logits, y, weights):
    """Weighted sum of the K-1 binary cross-entropies. `weights` re-balances the grades."""
    t = coral_targets(y, N_GRADES)
    per = nn.functional.binary_cross_entropy_with_logits(logits, t, reduction="none")
    return (per * weights[y].unsqueeze(1)).mean()


def grades_from_logits(logits):
    """Predicted grade = how many thresholds are passed."""
    return (torch.sigmoid(logits) > 0.5).sum(1)


class DS(Dataset):
    def __init__(self, X, y, tf):
        self.X, self.y, self.tf = X, y, tf

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        im = self.X[i]
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        return self.tf(np.ascontiguousarray(im)), int(self.y[i])


train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.15, 0.03)], p=0.7),
    transforms.RandomRotation(20),
    transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])
eval_tf = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
])


def load_split(split):
    DataClass = getattr(medmnist, INFO[DATASET]["python_class"])
    ds = DataClass(split=split, download=True, size=SIZE, root=DATA_ROOT)
    return ds.imgs, ds.labels.astype(np.int64).reshape(-1)


def ordinal_report(y, pred, tag):
    """The metrics an ordinal task is actually judged on."""
    d = np.abs(y - pred)
    n = len(y)
    out = {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "off_by_one_or_better": round(float((d <= 1).mean()), 4),
        "mean_absolute_error": round(float(d.mean()), 4),
        "quadratic_weighted_kappa": round(
            float(cohen_kappa_score(y, pred, weights="quadratic")), 4),
        "referable_sensitivity": None,
        "referable_specificity": None,
        "error_histogram": {int(k): int((d == k).sum()) for k in range(N_GRADES)},
    }
    yb, pb = (y >= REFERABLE).astype(int), (pred >= REFERABLE).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    out["referable_sensitivity"] = round(float(tp / max(tp + fn, 1)), 4)
    out["referable_specificity"] = round(float(tn / max(tn + fp, 1)), 4)
    out["referable_caught"] = "%d/%d" % (int(tp), int(tp + fn))
    log("  [%s] acc=%.4f  QWK=%.4f  MAE=%.3f  within1=%.4f  referable sens=%.4f (%s)"
        % (tag, out["accuracy"], out["quadratic_weighted_kappa"],
           out["mean_absolute_error"], out["off_by_one_or_better"],
           out["referable_sensitivity"], out["referable_caught"]))
    return out


def main():
    t0 = time.time()
    log("\n[%s] %s size=%d device=%s  ORDINAL (CORAL) head" % (KEY, DATASET, SIZE, DEVICE))
    Xtr, ytr = load_split("train"); Xva, yva = load_split("val"); Xte, yte = load_split("test")
    log("[%s] train=%d val=%d test=%d | test grades=%s"
        % (KEY, len(ytr), len(yva), len(yte), np.bincount(yte, minlength=N_GRADES).tolist()))

    tl = DataLoader(DS(Xtr, ytr, train_tf), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(DS(Xva, yva, eval_tf), batch_size=64, num_workers=0)
    el = DataLoader(DS(Xte, yte, eval_tf), batch_size=64, num_workers=0)

    counts = np.bincount(ytr, minlength=N_GRADES)
    w = counts.sum() / (N_GRADES * np.maximum(counts, 1))
    weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    log("[%s] grade counts=%s  class weights=%s"
        % (KEY, counts.tolist(), np.round(w, 3).tolist()))

    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = net.fc.in_features
    net.fc = CoralHead(in_f, N_GRADES, DROPOUT)
    net = net.to(DEVICE)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    def freeze(frozen):
        for name, p in net.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = not frozen

    @torch.no_grad()
    def collect(loader, tta=False):
        net.eval(); ys, ps = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                lg = net(xb).float()
                if tta:
                    lg = (lg + net(torch.flip(xb, dims=[3])).float()) / 2
            ps.append(grades_from_logits(lg).cpu().numpy()); ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    freeze(True)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                            lr=HEAD_LR, weight_decay=1e-4)
    stage, sched = "A(head)", None
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(EPOCHS):
        if ep == WARMUP:
            freeze(False)
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, EPOCHS - WARMUP))
            stage = "B(full)"
        net.train(); tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = coral_loss(net(xb).float(), yb, weights)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(xb)
        if sched:
            sched.step()
        yv, pv = collect(vl)
        vacc = accuracy_score(yv, pv)
        vqwk = cohen_kappa_score(yv, pv, weights="quadratic")
        hist.append({"epoch": ep + 1, "stage": stage, "train_loss": round(tot / len(ytr), 4),
                     "val_acc": round(float(vacc), 4), "val_qwk": round(float(vqwk), 4)})
        log("  ep %2d/%d [%s] loss=%.4f val_acc=%.4f val_qwk=%.4f"
            % (ep + 1, EPOCHS, stage, tot / len(ytr), vacc, vqwk))
        if not np.isfinite(tot):
            raise SystemExit("[FATAL] non-finite loss at epoch %d — retry with AMP=0" % (ep + 1))
        # Selected on val ACCURACY so the comparison with retina_v2 is like-for-like.
        if vacc > best + 1e-4:
            best, best_state, bad = vacc, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                log("  [early stop] patience %d" % PATIENCE); break
    net.load_state_dict(best_state)

    # TTA kept only if it helps on VAL, exactly as the v2 recipe does.
    yv, pv_no = collect(vl); _, pv_tta = collect(vl, tta=True)
    use_tta = accuracy_score(yv, pv_tta) > accuracy_score(yv, pv_no)
    log("[%s] val no-TTA=%.4f TTA=%.4f -> use_tta=%s"
        % (KEY, accuracy_score(yv, pv_no), accuracy_score(yv, pv_tta), use_tta))

    # Did the learned biases end up sorted? If yes, cumulative probabilities are monotone for
    # every input. CORAL guarantees the ordering is CONSISTENT across inputs, not that it is
    # descending, so this is checked and reported rather than assumed.
    b = net.fc.bias.detach().cpu().numpy()
    b_sorted = bool(np.all(b[:-1] >= b[1:]))
    log("[%s] learned thresholds b=%s  descending=%s"
        % (KEY, np.round(b, 3).tolist(), b_sorted))

    yt, pt = collect(el, tta=use_tta)
    ytr_e, ptr_e = collect(DataLoader(DS(Xtr, ytr, eval_tf), batch_size=64, num_workers=0))
    log("\n[report]")
    log(classification_report(yt, pt, labels=list(range(N_GRADES)),
                              target_names=[str(i) for i in range(N_GRADES)], zero_division=0))
    test_m = ordinal_report(yt, pt, "test")
    train_acc = float(accuracy_score(ytr_e, ptr_e))

    # The baseline this must beat, read from its own metrics file — no hardcoded numbers.
    base_path = os.path.join(MODEL_DIR, "retina_v2_metrics.json")
    base = json.load(open(base_path, encoding="utf-8")) if os.path.exists(base_path) else {}
    base_cm = np.array(base.get("confusion_matrix", []))
    base_m = None
    if base_cm.size:
        yb, pb = [], []
        for i in range(base_cm.shape[0]):
            for j in range(base_cm.shape[1]):
                yb += [i] * int(base_cm[i, j]); pb += [j] * int(base_cm[i, j])
        base_m = ordinal_report(np.array(yb), np.array(pb), "retina_v2 baseline")

    metrics = {
        "model": "%s_resnet18_coral" % KEY, "dataset_key": KEY, "medmnist": DATASET,
        "task": "ordinal 5-grade diabetic retinopathy (CORAL head)",
        "head": "CORAL: shared weight vector + %d free biases -> monotone cumulative probs"
                % (N_GRADES - 1),
        "what_changed_vs_retina_v2": ("ONLY the head and the loss. Same ResNet-18 backbone, "
                                      "same two-stage schedule, augmentation, optimiser, seed "
                                      "and splits, so the difference is attributable to the "
                                      "ordinal formulation."),
        "input_size": SIZE, "n_classes": N_GRADES,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "test_accuracy": test_m["accuracy"], "train_accuracy": round(train_acc, 4),
        "overfitting_gap": round(train_acc - test_m["accuracy"], 4),
        "test_ordinal_metrics": test_m,
        "retina_v2_baseline_same_metrics": base_m,
        "delta_accuracy_vs_retina_v2": (None if not base_m else
                                        round(test_m["accuracy"] - base_m["accuracy"], 4)),
        "delta_qwk_vs_retina_v2": (None if not base_m else
                                   round(test_m["quadratic_weighted_kappa"]
                                         - base_m["quadratic_weighted_kappa"], 4)),
        "tta_used": bool(use_tta),
        "coral_thresholds": [round(float(v), 4) for v in b],
        "coral_thresholds_descending": b_sorted,
        "confusion_matrix": confusion_matrix(yt, pt, labels=list(range(N_GRADES))).tolist(),
        "epoch_history": hist,
        "test_split": "full official MedMNIST test split (never subsampled, never tuned on)",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1), "device": DEVICE, "seed": SEED,
    }
    with open(os.path.join(MODEL_DIR, "%s_metrics.json" % KEY), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    torch.save({"state_dict": net.state_dict(), "size": SIZE,
                "classes": [str(i) for i in range(N_GRADES)],
                "mean": IM_MEAN, "std": IM_STD, "dropout": DROPOUT,
                "head": "coral", "n_grades": N_GRADES, "tta": bool(use_tta),
                "medmnist": DATASET, "binary_task": False, "binary_positive": None,
                "arch": "resnet18_coral"},
               os.path.join(MODEL_DIR, "%s.pt" % KEY))

    if base_m:
        log("\n[RESULT] %s acc=%.4f (retina_v2 %.4f, delta %+.4f) | QWK=%.4f (%.4f, %+.4f)"
            % (KEY, test_m["accuracy"], base_m["accuracy"],
               test_m["accuracy"] - base_m["accuracy"],
               test_m["quadratic_weighted_kappa"], base_m["quadratic_weighted_kappa"],
               test_m["quadratic_weighted_kappa"] - base_m["quadratic_weighted_kappa"]))
    else:
        log("\n[RESULT] %s test_acc=%.4f" % (KEY, test_m["accuracy"]))
    log("RETINA_ORD_DONE")


if __name__ == "__main__":
    main()
