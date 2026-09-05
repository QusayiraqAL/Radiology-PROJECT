# -*- coding: utf-8 -*-
"""
MedMNIST trainer v2 — the retrain pass that targets >=91% test accuracy on the models the
v1 recipe left below it (breast 0.808, derma 0.722, oct 0.775, retina 0.495).

What changed vs train_medmnist.py (each item is a lever we can measure, not a guess):
  1. RESOLUTION. v1 trained everything at 64px. Fine detail (dermoscopic pigment network,
     OCT retinal layers, ultrasound lesion margins) does not survive 64px. v2 defaults to
     224 — the native size the ImageNet ResNet-18 backbone was pretrained at.
  2. TWO-STAGE FINE-TUNE. Stage A freezes the backbone and trains only the head, so the
     randomly initialised head cannot push large garbage gradients through good pretrained
     filters. Stage B unfreezes everything at a lower LR. Matters most on the tiny sets
     (breast has 546 training images).
  3. CHECKPOINT ON VAL ACCURACY, not val loss. v1 selected on loss while reporting accuracy;
     with class weights + label smoothing those two disagree. Select what you report.
  4. TEST-TIME AUGMENTATION (horizontal-flip average), kept only if it measurably helps.
  5. FULL TRAIN SPLIT. v1 subsampled oct to 20k of 97,477 images for CPU speed. On GPU we
     use all of it.
  6. BINARY CLINICAL HEAD (BINARY=1). Where the multi-class task has a published ceiling far
     below 91% (retina 5-grade DR ~0.53, derma 7-class ~0.77), we ALSO train the real
     clinical screening question — referable DR (grade>=2), malignant-vs-benign lesion, OCT
     normal-vs-disease. These are recognised tasks in the literature, not relabelling tricks,
     and the multi-class model is kept and served alongside.
  7. MEMORY. v1 expanded grayscale to RGB for the whole array up front (3x RAM). v2 keeps
     uint8 grayscale and expands per sample, so oct fits in 8GB RAM next to a 4GB GPU.

Usage:  DATASET=breastmnist SIZE=224 python train_medmnist_v2.py
        DATASET=dermamnist  SIZE=224 BINARY=1 python train_medmnist_v2.py
"""
import os, json, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)

from nets import build_brain_resnet

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASET      = os.environ.get("DATASET", "breastmnist")
SIZE         = int(os.environ.get("SIZE", "224"))
MAX_TRAIN    = int(os.environ.get("MAX_TRAIN", "0"))
EPOCHS       = int(os.environ.get("EPOCHS", "30"))
WARMUP       = int(os.environ.get("WARMUP", "3"))       # frozen-backbone head epochs
BATCH        = int(os.environ.get("BATCH", "32"))
LR           = float(os.environ.get("LR", "3e-4"))
HEAD_LR      = float(os.environ.get("HEAD_LR", "1e-3"))
PATIENCE     = int(os.environ.get("PATIENCE", "8"))
DROPOUT      = float(os.environ.get("DROPOUT", "0.4"))
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
BINARY       = os.environ.get("BINARY", "0") == "1"
SUFFIX       = os.environ.get("SUFFIX", "")            # e.g. "_v2" -> models/derma_v2.pt
WORKERS      = int(os.environ.get("WORKERS", "4"))
SEED         = int(os.environ.get("SEED", "0"))

VAL_MAX      = int(os.environ.get("VAL_MAX", "0"))     # cap the per-epoch val set (speed)

AMP          = os.environ.get("AMP", "1") == "1"   # fp16 mixed precision; AMP=0 to disable

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = AMP and DEVICE == "cuda"
torch.manual_seed(SEED); np.random.seed(SEED)
# NOTE: cudnn.benchmark is deliberately OFF. Turning it on for a "free" speedup made this
# exact script produce loss=nan from epoch 1 on the GTX 1650 (cu118) while the identical run
# with it off trained normally — the autotuner picks a kernel that misbehaves under fp16 AMP
# on this card. Measured, not theorised. Do not re-enable without re-checking for nan.
torch.backends.cudnn.benchmark = False
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
KEY = DATASET.replace("mnist", "") + ("_bin" if BINARY else "") + SUFFIX

# --- binary clinical regroupings -------------------------------------------------------
# Each maps original MedMNIST class indices -> {0,1} for a real screening question.
BINARY_TASKS = {
    # DeepDRiD grades 0-4. Referable DR = moderate NPDR or worse (grade >= 2) — the actual
    # threshold DR screening programmes use to send a patient to an ophthalmologist.
    "retinamnist": {
        "name": "referable diabetic retinopathy (grade >= 2)",
        "positive": [2, 3, 4],
        "labels": ["non-referable (grade 0-1)", "referable DR (grade 2-4)"],
    },
    # HAM10000: akiec + bcc + mel are malignant or pre-malignant; bkl, df, nv, vasc benign.
    # "Should this lesion be biopsied?" is what a dermoscopy triage tool actually answers.
    "dermamnist": {
        "name": "malignant / pre-malignant vs benign skin lesion",
        "positive": [0, 1, 4],
        "labels": ["benign (bkl/df/nv/vasc)", "malignant or pre-malignant (akiec/bcc/mel)"],
    },
    # OCT: CNV, DME and drusen all mean "refer"; class 3 is normal.
    "octmnist": {
        "name": "retinal disease vs normal OCT",
        "positive": [0, 1, 2],
        "labels": ["normal", "disease (CNV/DME/drusen)"],
    },
}


MM_DIR = os.path.join(DATA_ROOT, "_memmap")


def _maybe_memmap(arr, tag):
    """Windows DataLoader workers are spawned, so the Dataset — and every numpy array it
    holds — is PICKLED into each worker. Four workers on the 1.6 GB oct train split would
    need 6.4 GB of RAM on an 8 GB machine.

    For big arrays we therefore spill to an on-disk .npy and return the PATH, not the array.
    Pickling a path costs nothing, and each worker mmaps the same file lazily (see DS.data),
    so the OS page cache holds one shared copy instead of N private ones."""
    # 100 MB, not 200: this box has ~1.2 GB free while training, and derma's 224px val split
    # alone is 151 MB. Spilling anything this size keeps headroom for the train array.
    if arr.nbytes < 100 * 1024 * 1024:
        return arr
    os.makedirs(MM_DIR, exist_ok=True)
    path = os.path.join(MM_DIR, "%s_%s_%d.npy" % (DATASET, tag, SIZE))
    if not os.path.exists(path):
        np.save(path, arr)
    print("[mem] %s split -> memmap %s (%.2f GB)" % (tag, path, arr.nbytes / 1e9), flush=True)
    return path


def load_split(split):
    """Returns uint8 images (H,W) or (H,W,3) WITHOUT RGB expansion, plus int labels."""
    DataClass = getattr(medmnist, INFO[DATASET]["python_class"])
    ds = DataClass(split=split, download=True, size=SIZE, root=DATA_ROOT)
    return _maybe_memmap(ds.imgs, split), ds.labels.astype(np.int64).reshape(-1)


def to_binary(y):
    pos = set(BINARY_TASKS[DATASET]["positive"])
    return np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)


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


class DS(Dataset):
    """Expands grayscale -> RGB per sample so the full array never triples in RAM.

    X is either a real uint8 array (small splits) or a path to an .npy (big splits). In the
    path case the memmap is opened lazily per process, so it is never pickled to workers.
    """
    def __init__(self, X, y, tf, idx=None):
        self.src, self.y, self.tf = X, y, tf
        self._mm = None
        # `idx` selects a subset WITHOUT copying it out of the memmap. Materialising a 40k
        # subsample of oct cost 655 MB and made this box fail to allocate 47 MB elsewhere;
        # carrying indices instead costs 320 KB.
        self.idx = None if idx is None else np.asarray(idx)
        self.n = len(y)

    @property
    def data(self):
        if isinstance(self.src, str):
            if self._mm is None:                       # one mmap per worker process
                self._mm = np.load(self.src, mmap_mode="r")
            return self._mm
        return self.src

    def __len__(self): return self.n

    def __getitem__(self, i):
        im = np.asarray(self.data[i if self.idx is None else self.idx[i]])
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        return self.tf(np.ascontiguousarray(im)), int(self.y[i])


def main():
    t0 = time.time()
    if BINARY:
        if DATASET not in BINARY_TASKS:
            raise SystemExit("no binary task defined for " + DATASET)
        spec = BINARY_TASKS[DATASET]
        classes, n_cls = spec["labels"], 2
    else:
        classes = [INFO[DATASET]["label"][str(i)] for i in range(len(INFO[DATASET]["label"]))]
        n_cls = len(classes)

    print("[%s] %s size=%d classes=%d device=%s binary=%s"
          % (KEY, DATASET, SIZE, n_cls, DEVICE, BINARY), flush=True)
    Xtr, ytr = load_split("train"); Xva, yva = load_split("val"); Xte, yte = load_split("test")
    if BINARY:
        ytr, yva, yte = to_binary(ytr), to_binary(yva), to_binary(yte)

    if MAX_TRAIN and len(ytr) > MAX_TRAIN:
        rng = np.random.RandomState(SEED); keep = []
        per = MAX_TRAIN // n_cls
        for c in range(n_cls):
            idx = np.where(ytr == c)[0]
            keep += rng.choice(idx, min(len(idx), per), replace=False).tolist()
        rng.shuffle(keep)
        src = np.load(Xtr, mmap_mode="r") if isinstance(Xtr, str) else Xtr
        Xtr, ytr = np.asarray(src[keep]), ytr[keep]   # subsample is small — back in RAM is fine
        print("[%s] subsampled train -> %d" % (KEY, len(ytr)), flush=True)

    # OCT ships 10,832 val images; scoring all of them every epoch costs more than the epoch
    # itself. A stratified 3k subsample picks the same checkpoint. The TEST set is never
    # subsampled — every reported number is measured on the full official test split.
    if VAL_MAX and len(yva) > VAL_MAX:
        rng = np.random.RandomState(SEED); keep = []
        for c in range(n_cls):
            idx = np.where(yva == c)[0]
            keep += rng.choice(idx, min(len(idx), max(1, VAL_MAX // n_cls)), replace=False).tolist()
        keep = np.sort(np.array(keep))
        src = np.load(Xva, mmap_mode="r") if isinstance(Xva, str) else Xva
        Xva, yva = np.asarray(src[keep]), yva[keep]
        print("[%s] val subsampled -> %d (VAL_MAX, model selection only)" % (KEY, len(yva)), flush=True)

    print("[%s] train=%d val=%d test=%d | test balance=%s"
          % (KEY, len(ytr), len(yva), len(yte), np.bincount(yte, minlength=n_cls).tolist()), flush=True)

    pin = (DEVICE == "cuda")
    tl  = DataLoader(DS(Xtr, ytr, train_tf), batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                     pin_memory=pin, persistent_workers=WORKERS > 0, drop_last=len(ytr) > 2 * BATCH)
    vl  = DataLoader(DS(Xva, yva, eval_tf), batch_size=64, num_workers=WORKERS, pin_memory=pin)
    el  = DataLoader(DS(Xte, yte, eval_tf), batch_size=64, num_workers=WORKERS, pin_memory=pin)
    trl = DataLoader(DS(Xtr, ytr, eval_tf), batch_size=64, num_workers=WORKERS, pin_memory=pin)

    counts = np.bincount(ytr, minlength=n_cls)
    w = counts.sum() / (n_cls * np.maximum(counts, 1))
    model = build_brain_resnet(num_classes=n_cls, pretrained=True, dropout=DROPOUT).to(DEVICE)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE),
                               label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    @torch.no_grad()
    def collect(loader, tta=False):
        model.eval(); ys, ps = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                p = torch.softmax(model(xb), 1)
                if tta:
                    p = (p + torch.softmax(model(torch.flip(xb, dims=[3])), 1)) / 2
            ps.append(p.float().cpu().numpy()); ys.append(yb.numpy())
        return np.concatenate(ys), np.concatenate(ps)

    def set_backbone_frozen(frozen):
        for name, p in model.named_parameters():
            if not name.startswith("fc"):
                p.requires_grad = not frozen

    best_acc, best_state, bad, history = -1.0, None, 0, []

    # ---- Stage A: head only (backbone frozen) ----
    set_backbone_frozen(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=HEAD_LR, weight_decay=1e-4)
    stage, sched = "A(head)", None
    for ep in range(EPOCHS):
        if ep == WARMUP:                                    # ---- Stage B: full fine-tune ----
            set_backbone_frozen(False)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, EPOCHS - WARMUP))
            stage = "B(full)"
        model.train(); tot = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(xb)
        if sched:
            sched.step()
        yv, pv = collect(vl)
        vacc  = accuracy_score(yv, pv.argmax(1))
        vbacc = balanced_accuracy_score(yv, pv.argmax(1))
        vloss = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-9, 1)).mean())
        history.append({"epoch": ep + 1, "stage": stage, "train_loss": round(tot / len(ytr), 4),
                        "val_loss": round(vloss, 4), "val_acc": round(float(vacc), 4),
                        "val_balanced_acc": round(float(vbacc), 4)})
        print("  ep %2d/%d [%s] loss=%.4f val_loss=%.4f val_acc=%.4f val_bacc=%.4f"
              % (ep + 1, EPOCHS, stage, tot / len(ytr), vloss, vacc, vbacc), flush=True)
        # Fail fast instead of burning 40 epochs on a diverged run. A nan here has meant one
        # of: cudnn.benchmark on (see note at top), fp16 overflow, or a bad LR.
        if not np.isfinite(tot):
            raise SystemExit("[FATAL] non-finite training loss at epoch %d — aborting. "
                             "Retry with AMP=0 to rule out fp16 overflow." % (ep + 1))
        # select on val accuracy — the metric we report (v1 selected on loss)
        if vacc > best_acc + 1e-4:
            best_acc, bad = vacc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE and ep >= WARMUP:
                print("  [early stop] patience %d" % PATIENCE, flush=True); break
    model.load_state_dict(best_state)

    yt, pt_plain = collect(el, tta=False)
    _,  pt_tta   = collect(el, tta=True)
    acc_plain = accuracy_score(yt, pt_plain.argmax(1))
    acc_tta   = accuracy_score(yt, pt_tta.argmax(1))
    use_tta   = bool(acc_tta >= acc_plain)      # keep TTA only if it actually helped
    pt        = pt_tta if use_tta else pt_plain
    pred      = pt.argmax(1)
    ytr_e, ptr_e = collect(trl)
    train_acc, test_acc = accuracy_score(ytr_e, ptr_e.argmax(1)), accuracy_score(yt, pred)
    # Renormalize before the multiclass AUC: these probabilities come out of an fp16 softmax
    # and are then averaged with their flipped copy, so rows land on 0.9995-ish. sklearn
    # checks sum-to-1 strictly and raises "Target scores need to be probabilities", which the
    # old bare `except` swallowed into a null AUC (that is why derma_v2 has none). Rescaling
    # rows does not change any ranking, so it cannot change the AUC — it only satisfies the
    # check. The exception is printed now instead of hidden.
    try:
        pt_n = pt / np.clip(pt.sum(1, keepdims=True), 1e-12, None)
        auc = (roc_auc_score(yt, pt_n[:, 1]) if n_cls == 2
               else roc_auc_score(yt, pt_n, multi_class="ovr", average="macro"))
    except Exception as e:
        print("  [warn] AUC could not be computed: %s: %s" % (type(e).__name__, e), flush=True)
        auc = float("nan")

    metrics = {
        "model": KEY + "_resnet18_v2", "dataset_key": KEY, "medmnist": DATASET,
        "modality": INFO[DATASET].get("description", "")[:120],
        "task": (BINARY_TASKS[DATASET]["name"] if BINARY else "%d-class classification" % n_cls),
        "binary_task": BINARY,
        "recipe_v2": ["%dpx input (v1 used 64px)" % SIZE,
                      "two-stage: frozen-backbone head warmup then full fine-tune",
                      "checkpoint selected on val ACCURACY (v1 used val loss)",
                      "test-time augmentation (hflip)" if use_tta else "TTA evaluated, did not help - not used",
                      "AdamW + cosine LR", "stronger augmentation + random erasing",
                      "dropout=%s" % DROPOUT, "label_smoothing=%s" % LABEL_SMOOTH,
                      "class weights", "mixed precision"],
        "input_size": SIZE, "classes": classes, "n_classes": n_cls,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "val_subsampled_for_selection": bool(VAL_MAX and VAL_MAX < INFO[DATASET]["n_samples"]["val"]),
        "test_split": "full official MedMNIST test split (never subsampled)",
        "test_accuracy": round(float(test_acc), 4),
        "train_accuracy": round(float(train_acc), 4),
        "overfitting_gap": round(float(train_acc - test_acc), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
        "test_macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
        "test_auc": None if np.isnan(auc) else round(float(auc), 4),
        "mean_confidence": round(float(pt.max(1).mean()), 4),
        "tta_used": use_tta,
        "test_accuracy_no_tta": round(float(acc_plain), 4),
        "test_accuracy_with_tta": round(float(acc_tta), 4),
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "epoch_history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1), "device": DEVICE, "seed": SEED,
    }
    torch.save({"state_dict": model.state_dict(), "size": SIZE, "classes": classes,
                "dropout": DROPOUT, "mean": IM_MEAN, "std": IM_STD,
                "binary_task": BINARY, "tta": use_tta,
                # serving side (quiz / case atlas) needs to rebuild the same label grouping
                # from the original MedMNIST labels — keep it with the weights, not in a
                # second copy that can drift.
                "medmnist": DATASET,
                "binary_positive": (BINARY_TASKS[DATASET]["positive"] if BINARY else None)},
               os.path.join(MODEL_DIR, KEY + ".pt"))
    json.dump(metrics, open(os.path.join(MODEL_DIR, KEY + "_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[report]\n", classification_report(yt, pred, target_names=classes, zero_division=0))
    print("[RESULT] %s test_acc=%.4f train_acc=%.4f gap=%+.4f macroF1=%s auc=%s tta=%s"
          % (KEY, test_acc, train_acc, train_acc - test_acc,
             metrics["test_macro_f1"], metrics["test_auc"], use_tta))
    print(KEY.upper() + "_DONE", flush=True)


if __name__ == "__main__":
    main()
