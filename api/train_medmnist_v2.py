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

from nets import build_medmnist_backbone, ARCH_MAX_BATCH
import preproc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
MODEL_DIR = os.path.join(HERE, "models")
os.makedirs(DATA_ROOT, exist_ok=True)

DATASET      = os.environ.get("DATASET", "breastmnist")
SIZE         = int(os.environ.get("SIZE", "224"))
MAX_TRAIN    = int(os.environ.get("MAX_TRAIN", "0"))
EPOCHS       = int(os.environ.get("EPOCHS", "30"))
WARMUP       = int(os.environ.get("WARMUP", "3"))       # frozen-backbone head epochs
ARCH         = os.environ.get("ARCH", "resnet18")     # see nets.MEDMNIST_ARCHS
# Default the batch to the largest one measured to fit this 4 GB card at 224px fp32
# (nets.ARCH_MAX_BATCH). An explicit BATCH= still wins.
BATCH        = int(os.environ.get("BATCH", str(ARCH_MAX_BATCH.get(ARCH, 32))))
LR           = float(os.environ.get("LR", "3e-4"))
HEAD_LR      = float(os.environ.get("HEAD_LR", "1e-3"))
PATIENCE     = int(os.environ.get("PATIENCE", "8"))
DROPOUT      = float(os.environ.get("DROPOUT", "0.4"))
LABEL_SMOOTH = float(os.environ.get("LABEL_SMOOTH", "0.05"))
BINARY       = os.environ.get("BINARY", "0") == "1"
SUFFIX       = os.environ.get("SUFFIX", "")            # e.g. "_v2" -> models/derma_v2.pt
WORKERS      = int(os.environ.get("WORKERS", "4"))
SEED         = int(os.environ.get("SEED", "0"))

# --- regularisation levers (0 = off, which reproduces the pre-session-6 recipe) -------
# retina_v2 measured train 0.8250 vs test 0.6075 — a +0.2175 gap on 1080 training
# images. mixup/cutmix attack exactly that gap and cost no extra memory, which matters
# on a box that has already lost three runs to MemoryError.
MIXUP        = float(os.environ.get("MIXUP", "0"))    # Beta(a,a) alpha for mixup

# Fixed preprocessing applied ONCE per split, before any augmentation. See preproc.py.
# "ben_graham" is the Kaggle-DR unsharp mask; "clahe" is adaptive histogram equalisation.
# Applied to the array rather than per sample because it is deterministic - CLAHE costs
# 171 ms an image, which would add ~185 s to every epoch of a 9 s retina epoch if it ran
# inside __getitem__. It is written into the checkpoint so serving applies the same thing.
PREPROC      = os.environ.get("PREPROC", "none")

# --- K-fold cross-validation over train+val ---------------------------------------------
# Session 6 kept hitting the same wall: breast has 78 validation images and retina 120, and
# at that size val cannot rank two models. On breast it ranked three candidates in exactly
# the REVERSE of their test order, one image apart each (step 45), so no promotion there was
# defensible. Pooling train+val and rotating the held-out fold turns a 78-image selection
# signal into a 624-image one, which is the whole reason the technique exists.
#
# KFOLD=5 FOLD=k trains on 4/5 of train+val and validates on the remaining fifth, stratified,
# with a fixed seed so the folds are identical across every configuration compared. The
# official TEST split is untouched and is still only ever measured, never selected on.
# --- augmentation policy ------------------------------------------------------------------
# "full" is the v2 recipe: RandomResizedCrop(0.7-1.0) + h-flip + v-flip + 20 deg rotation +
# colour jitter + random erasing. It assumes the label is invariant to position and
# orientation. That assumption is FALSE for organc, whose eleven classes include
# kidney-left/kidney-right, lung-left/lung-right and femur-left/femur-right - a horizontal
# flip literally relabels those images.
#
# Measured on organc_eb0 (efficientnet_b0 @224, full augmentation): 428 of its 810 test errors
# (52.8%) are left/right swaps, and it scores 0.9014 against the served 64px v1's 0.9419.
# TTA is not the cause - val 0.9741 vs 0.9745, test 0.9023 vs 0.9014, i.e. +/-0.001. The
# damage is done during training.
#
# "geometry_safe" keeps the photometric augmentation and drops everything that moves or
# mirrors anatomy: no flips, rotation limited to 7 degrees, and a gentle crop.
AUG          = os.environ.get("AUG", "full")     # "full" | "geometry_safe"

# Which TTA view sets to put in front of val. The default ["none", "flip"] is exactly the
# historical choice - identity vs identity+hflip - so an untouched run reproduces its old
# number. TTA_SET=multi adds the 5-view set from the Kaggle DR notebook; TTA_SET=safe offers
# the mirror-free set instead. More candidates means more freedom to fit val noise, which is
# fine on a 10k val (path) and not fine on 78 (breast) - hence a knob, not a default.
TTA_SET      = os.environ.get("TTA_SET", "default")

# --- what "best epoch" means --------------------------------------------------------------
# v1 selected on val LOSS while reporting accuracy; v2 fixed that to val ACCURACY (step 5,
# "select what you report"). But accuracy is the wrong target for an imbalanced screening
# task: derma_bin's val is 80% benign, so a checkpoint that gets better at benign outscores
# one that catches more malignancies. That is the same trade the step-14 guard refuses at the
# THRESHOLD, applied one level up - at which epoch's weights we keep.
#
# The Project-Melanin notebook in code benifit/ selects on recall for exactly this reason
# (`best_recall = 0.0; if val_recall > best_recall`). Offered here as a knob and measured,
# not switched on by assumption:
#   acc     val accuracy            - the v2 default, unchanged
#   bacc    val balanced accuracy   - every class weighted equally
#   recall  val recall of the disease class (binary tasks only)
SELECT_ON    = os.environ.get("SELECT_ON", "acc")

# --- loss --------------------------------------------------------------------------------
# "ce" is weighted cross-entropy with label smoothing - the v2 recipe, unchanged.
# "focal" adds the (1-p_t)^gamma factor from the ISIC notebook in code benifit/: it scales
# each example's loss DOWN as the model gets more confident about it, so gradient budget
# moves to the examples still being got wrong. Class weights already re-weight by class;
# focal re-weights by difficulty, which is a different axis and composes with them.
# gamma=0 makes focal identical to plain weighted CE, which is the regression test.
LOSS         = os.environ.get("LOSS", "ce")          # "ce" | "focal"
FOCAL_GAMMA  = float(os.environ.get("FOCAL_GAMMA", "2.0"))
KFOLD        = int(os.environ.get("KFOLD", "0"))
FOLD         = int(os.environ.get("FOLD", "0"))
CUTMIX       = float(os.environ.get("CUTMIX", "0"))   # Beta(a,a) alpha for cutmix

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
    # The cache key MUST carry every transform baked into the bytes. It used to be
    # (dataset, split, size) only, so the first PREPROC run reused the raw array that earlier
    # runs had spilled: retina's train split is 162 MB and gets memmapped, while its 18 MB val
    # and 60 MB test stay in RAM and DID get preprocessed. The model therefore trained on raw
    # images and was scored on ben_graham ones - train 0.7056, test 0.2500 (step 73).
    tags = [DATASET, tag, str(SIZE)]
    if PREPROC and PREPROC != "none":
        tags.append(PREPROC)
    path = os.path.join(MM_DIR, "_".join(tags) + ".npy")
    if not os.path.exists(path):
        np.save(path, arr)
    print("[mem] %s split -> memmap %s (%.2f GB)" % (tag, path, arr.nbytes / 1e9), flush=True)
    return path


def load_split(split):
    """Returns uint8 images (H,W) or (H,W,3) WITHOUT RGB expansion, plus int labels."""
    DataClass = getattr(medmnist, INFO[DATASET]["python_class"])
    ds = DataClass(split=split, download=True, size=SIZE, root=DATA_ROOT)
    imgs = ds.imgs
    if PREPROC and PREPROC != "none":
        t = time.time()
        # Expands grayscale to 3 channels as a side effect, which is fine: the Dataset does
        # that per sample anyway, and doing it here costs the same memory the transform needs.
        imgs = np.stack([preproc.apply(imgs[i], PREPROC) for i in range(len(imgs))])
        print("[preproc] %s %s -> %s in %.1fs" % (PREPROC, split, imgs.shape, time.time() - t),
              flush=True)
    return _maybe_memmap(imgs, split), ds.labels.astype(np.int64).reshape(-1)


def to_binary(y):
    pos = set(BINARY_TASKS[DATASET]["positive"])
    return np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)


def _build_train_tf(policy):
    photometric = [transforms.RandomApply([transforms.ColorJitter(0.25, 0.25, 0.15, 0.03)],
                                          p=0.7)]
    tail = [transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))]
    # Order matters and is preserved from the original v2 pipeline: the photometric jitter
    # runs BEFORE the rotation. Rotation leaves black corners, and jittering after it would
    # apply brightness/contrast to those corners instead of leaving them at zero. The first
    # version of this refactor had them swapped - a small difference, but it silently made
    # AUG=full something other than the recipe every v2 number was measured with.
    if policy == "geometry_safe":
        pre_rot = [transforms.RandomResizedCrop(SIZE, scale=(0.90, 1.0), ratio=(0.95, 1.05))]
        rot = [transforms.RandomRotation(7)]
    elif policy == "full":
        pre_rot = [transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
                   transforms.RandomHorizontalFlip(),
                   transforms.RandomVerticalFlip(p=0.2)]
        rot = [transforms.RandomRotation(20)]
    else:
        raise SystemExit("unknown AUG policy %r; expected full or geometry_safe" % policy)
    return transforms.Compose([transforms.ToPILImage()] + pre_rot + photometric + rot + tail)


train_tf = _build_train_tf(AUG)
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


def mix_batch(xb, yb, rng):
    """mixup / cutmix on one batch. Returns (x, y_a, y_b, lam).

    lam is the weight of y_a, so the caller's loss is
        lam * crit(out, y_a) + (1 - lam) * crit(out, y_b)
    which keeps the existing class weights and label smoothing intact — building a soft
    target matrix by hand would have silently dropped both.

    lam = 1.0 means "nothing was mixed", so a run with MIXUP=CUTMIX=0 takes the same path
    as before this was added and reproduces the earlier numbers exactly.
    """
    if MIXUP <= 0 and CUTMIX <= 0:
        return xb, yb, yb, 1.0
    use_cut = CUTMIX > 0 and (MIXUP <= 0 or rng.random() < 0.5)
    alpha = CUTMIX if use_cut else MIXUP
    lam = float(rng.beta(alpha, alpha))
    perm = torch.randperm(xb.size(0), device=xb.device)
    y_a, y_b = yb, yb[perm]
    if not use_cut:
        return lam * xb + (1.0 - lam) * xb[perm], y_a, y_b, lam
    # cutmix: paste a lam-proportional box from the shuffled batch
    _, _, H, W = xb.shape
    r = np.sqrt(1.0 - lam)
    ch, cw = int(H * r), int(W * r)
    cy, cx = rng.integers(H), rng.integers(W)
    y1, y2 = np.clip([cy - ch // 2, cy + ch // 2], 0, H)
    x1, x2 = np.clip([cx - cw // 2, cx + cw // 2], 0, W)
    xb = xb.clone()
    xb[:, :, y1:y2, x1:x2] = xb[perm][:, :, y1:y2, x1:x2]
    # recompute lam from the box actually pasted (clipping changes the area)
    lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(H * W))
    return xb, y_a, y_b, lam


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

    print("[%s] %s arch=%s size=%d batch=%d classes=%d device=%s binary=%s mixup=%s cutmix=%s preproc=%s"
          % (KEY, DATASET, ARCH, SIZE, BATCH, n_cls, DEVICE, BINARY, MIXUP, CUTMIX,
             PREPROC + "/" + AUG),
          flush=True)
    if KFOLD > 1 and not SUFFIX:
        raise SystemExit("[FATAL] KFOLD needs a distinct SUFFIX or fold %d will overwrite "
                         "the served checkpoint." % FOLD)
    Xtr, ytr = load_split("train"); Xva, yva = load_split("val"); Xte, yte = load_split("test")
    if BINARY:
        ytr, yva, yte = to_binary(ytr), to_binary(yva), to_binary(yte)

    if KFOLD > 1:
        # Merge train+val, then carve fold FOLD out as the new val. Materialising here is safe
        # for the small sets this is meant for (breast 624, retina 1200); it refuses anything
        # big rather than quietly spilling an 8 GB box.
        src_tr = np.load(Xtr, mmap_mode="r") if isinstance(Xtr, str) else Xtr
        src_va = np.load(Xva, mmap_mode="r") if isinstance(Xva, str) else Xva
        n_all = len(ytr) + len(yva)
        if n_all > 8000:
            raise SystemExit("[FATAL] KFOLD on %d images would materialise the whole pool in "
                             "RAM on an 8 GB box. Intended for the small splits." % n_all)
        X_all = np.concatenate([np.asarray(src_tr), np.asarray(src_va)])
        y_all = np.concatenate([ytr, yva])
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=12345)
        tr_idx, va_idx = list(skf.split(np.zeros(len(y_all)), y_all))[FOLD]
        Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
        Xva, yva = X_all[va_idx], y_all[va_idx]
        print("[kfold] %d-fold, fold %d: pooled %d -> train %d / val %d (seed 12345, stratified)"
              % (KFOLD, FOLD, n_all, len(ytr), len(yva)), flush=True)
        del X_all, y_all

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
    model, head_prefix = build_medmnist_backbone(ARCH, num_classes=n_cls, pretrained=True,
                                                 dropout=DROPOUT)
    model = model.to(DEVICE)
    _wt = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    if LOSS == "ce":
        crit = nn.CrossEntropyLoss(weight=_wt, label_smoothing=LABEL_SMOOTH)
    elif LOSS == "focal":
        _base = nn.CrossEntropyLoss(weight=_wt, label_smoothing=LABEL_SMOOTH, reduction="none")

        def crit(logits, target):
            """Weighted, label-smoothed CE scaled by (1 - p_t)^gamma.

            p_t is read from the softmax rather than from exp(-ce) as the reference notebook
            does: with label smoothing and class weights, exp(-ce) is NOT the probability of
            the true class, so that shortcut would silently use a wrong focal factor. At
            gamma=0 the factor is 1 and this reduces exactly to the "ce" branch, which is how
            the implementation is checked.
            """
            ce = _base(logits, target)
            pt = torch.softmax(logits.detach(), 1).gather(1, target.view(-1, 1)).squeeze(1)
            # Normalise by the SUM OF CLASS WEIGHTS, not by N. With a `weight` argument and
            # reduction="mean", CrossEntropyLoss computes sum(w_i * l_i) / sum(w_i) - not the
            # plain mean. Using .mean() here made gamma=0 disagree with the "ce" branch by a
            # factor of sum(w)/N, i.e. it silently rescaled the effective learning rate.
            return (((1.0 - pt) ** FOCAL_GAMMA) * ce).sum() / _wt[target].sum()
    else:
        raise SystemExit("LOSS must be ce or focal")
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
        # "fc" is only the ResNet head name; EfficientNet and ConvNeXt call theirs
        # "classifier". Hardcoding "fc" here would freeze the head and train the
        # backbone — the exact inverse of stage A — with no error to show for it.
        for name, p in model.named_parameters():
            if not name.startswith(head_prefix):
                p.requires_grad = not frozen

    best_acc, best_state, bad, history = -1.0, None, 0, []
    # One kept checkpoint per selection metric, all taken from THIS trajectory.
    # Three separate runs cannot answer "which epoch should we keep", because the runs are not
    # reproducible: with the same seed they agree exactly while the backbone is frozen and then
    # diverge from the first unfrozen epoch, by up to 0.0220 val accuracy - the same size as the
    # k-fold noise floor itself (TRAINING_LOG step 90). Selecting three ways from one trajectory
    # drops that term to zero, and costs one run instead of three.
    alt_best = {}       # metric -> {"score", "epoch", "state"}
    mix_rng = np.random.default_rng(SEED)   # own stream: mixing must not shift torch's

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
            xb, y_a, y_b, lam = mix_batch(xb, yb, mix_rng)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                out = model(xb)
                loss = (crit(out, y_a) if lam >= 1.0
                        else lam * crit(out, y_a) + (1.0 - lam) * crit(out, y_b))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(xb)
        if sched:
            sched.step()
        yv, pv = collect(vl)
        vacc  = accuracy_score(yv, pv.argmax(1))
        vbacc = balanced_accuracy_score(yv, pv.argmax(1))
        if SELECT_ON not in ("acc", "bacc", "recall"):
            raise SystemExit("SELECT_ON must be acc, bacc or recall")
        epoch_scores = {"acc": float(vacc), "bacc": float(vbacc)}
        if n_cls == 2:
            # Disease is index 1 for a relabelled binary head; breastmnist keeps the original
            # order where malignant is index 0 (session 3, step 14).
            _pos = 1 if BINARY else (0 if DATASET == "breastmnist" else 1)
            _p = pv.argmax(1); _m = (yv == _pos)
            epoch_scores["recall"] = float((_p[_m] == _pos).sum() / max(1, _m.sum()))
        elif SELECT_ON == "recall":
            raise SystemExit("SELECT_ON=recall needs a binary task")
        score = epoch_scores[SELECT_ON]
        # Same keep-rule as the primary below, applied to every metric independently. This only
        # records candidates - it never touches `score`, so it cannot move training or early stop.
        for _mn, _ms in epoch_scores.items():
            if _mn not in alt_best or _ms > alt_best[_mn]["score"] + 1e-4:
                alt_best[_mn] = {"score": _ms, "epoch": ep + 1,
                                 "state": {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}}
        vloss = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-9, 1)).mean())
        history.append({"epoch": ep + 1, "stage": stage, "train_loss": round(tot / len(ytr), 4),
                        "val_loss": round(vloss, 4), "val_acc": round(float(vacc), 4),
                        "val_balanced_acc": round(float(vbacc), 4),
                        "val_selection_score": round(float(score), 4)})
        print("  ep %2d/%d [%s] loss=%.4f val_loss=%.4f val_acc=%.4f val_bacc=%.4f"
              % (ep + 1, EPOCHS, stage, tot / len(ytr), vloss, vacc, vbacc), flush=True)
        # Fail fast instead of burning 40 epochs on a diverged run. A nan here has meant one
        # of: cudnn.benchmark on (see note at top), fp16 overflow, or a bad LR.
        if not np.isfinite(tot):
            raise SystemExit("[FATAL] non-finite training loss at epoch %d — aborting. "
                             "Retry with AMP=0 to rule out fp16 overflow." % (ep + 1))
        # select on whatever SELECT_ON names; "acc" reproduces the v2 behaviour exactly
        if score > best_acc + 1e-4:
            best_acc, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE and ep >= WARMUP:
                print("  [early stop] patience %d" % PATIENCE, flush=True); break
    model.load_state_dict(best_state)

    # ---- the TTA decision is made on VAL, never on test -------------------------------
    # This used to compare acc_plain vs acc_tta on the TEST split and keep whichever won.
    # That is test-set selection: it reports max(a, b) of two test numbers, so it can only
    # ever move the headline up. Measured cost of the old rule on the shipped models:
    #   retina_v2 +0.0275 (0.5800 -> 0.6075), oct_v2 +0.0050, oct_bin +0.0040,
    #   derma_v2 +0.0025, everything else +0.0000.
    # The project's own rule (session 3, step 21) already said "test is not touched to select
    # anything - even the TTA decision is made on val". The local trainer was not obeying it.
    # Ties keep the plain path: TTA doubles inference cost at serve time, and a tie does not
    # buy that.
    yv_t, pv_plain = collect(vl, tta=False)
    _,    pv_tta   = collect(vl, tta=True)
    val_plain = accuracy_score(yv_t, pv_plain.argmax(1))
    val_tta   = accuracy_score(yv_t, pv_tta.argmax(1))
    # Under geometry_safe the model was deliberately NOT taught mirror invariance, so
    # averaging a prediction with its mirror asks it about an image it was never trained on.
    # The flag is forced off rather than left to a val margin that can be two images wide.
    use_tta   = bool(val_tta > val_plain) and AUG != "geometry_safe"
    tta_name  = "flip" if use_tta else "none"

    # ---- optionally widen the candidate view sets, still choosing on val -------------------
    if TTA_SET != "default":
        cands = {"multi": ["none", "flip", "multi"],
                 "safe":  ["none", "safe"],
                 "all":   ["none", "flip", "multi", "safe"]}.get(TTA_SET)
        if cands is None:
            raise SystemExit("TTA_SET must be one of default, multi, safe, all")
        if AUG == "geometry_safe":
            # Drop every set containing a mirror: the labels are not mirror-invariant here.
            cands = [c for c in cands if "hflip" not in preproc.TTA_VIEWS[c]]
        scores = {}
        for c in cands:
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for xb, yb in vl:
                    xb = xb.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        p = preproc.tta_average(model, xb, preproc.TTA_VIEWS[c])
                    ps.append(p.float().cpu().numpy()); ys.append(yb.numpy())
            scores[c] = accuracy_score(np.concatenate(ys), np.concatenate(ps).argmax(1))
        # Ties go to the cheapest set: TTA costs one forward pass per view at serve time.
        tta_name = min(scores, key=lambda c: (-scores[c], len(preproc.TTA_VIEWS[c])))
        use_tta = tta_name != "none"
        print("  [tta] val by view set: %s -> %s"
              % (", ".join("%s=%.4f" % (c, scores[c]) for c in cands), tta_name), flush=True)
    tta_views = preproc.TTA_VIEWS[tta_name]
    print("  [tta] val_plain=%.4f val_tta=%.4f -> use_tta=%s (decided on val)"
          % (val_plain, val_tta, use_tta), flush=True)

    yt, pt_plain = collect(el, tta=False)
    _,  pt_tta   = collect(el, tta=True)
    acc_plain = accuracy_score(yt, pt_plain.argmax(1))
    acc_tta   = accuracy_score(yt, pt_tta.argmax(1))
    if tta_name in ("none", "flip"):
        pt = pt_tta if use_tta else pt_plain
    else:
        # A wider view set was chosen on val; score test through that same set.
        model.eval(); _ps = []
        with torch.no_grad():
            for xb, _yb in el:
                xb = xb.to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    _ps.append(preproc.tta_average(model, xb, tta_views).float().cpu().numpy())
        pt = np.concatenate(_ps)
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
        "model": KEY + "_" + ARCH + "_v2", "arch": ARCH, "preproc": PREPROC,
        "kfold": KFOLD or None, "fold": FOLD if KFOLD > 1 else None, "aug": AUG,
        "selected_on": SELECT_ON, "loss": LOSS,
        "focal_gamma": FOCAL_GAMMA if LOSS == "focal" else None,
        "dataset_key": KEY, "medmnist": DATASET,
        "modality": INFO[DATASET].get("description", "")[:120],
        "task": (BINARY_TASKS[DATASET]["name"] if BINARY else "%d-class classification" % n_cls),
        "binary_task": BINARY,
        "recipe_v2": ["%dpx input (v1 used 64px)" % SIZE,
                      "two-stage: frozen-backbone head warmup then full fine-tune",
                      "checkpoint selected on val ACCURACY (v1 used val loss)",
                      "test-time augmentation (hflip)" if use_tta else "TTA evaluated, did not help - not used",
                      "AdamW + cosine LR", "stronger augmentation + random erasing",
                      "dropout=%s" % DROPOUT, "label_smoothing=%s" % LABEL_SMOOTH,
                      "class weights", "mixed precision",
                      "backbone=%s" % ARCH]
                     + (["mixup alpha=%s" % MIXUP] if MIXUP > 0 else [])
                     + (["cutmix alpha=%s" % CUTMIX] if CUTMIX > 0 else []),
        "mixup_alpha": MIXUP, "cutmix_alpha": CUTMIX, "batch_size": BATCH,
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
        # Guarded the same way test_auc above is. An fp16 eval pass can hand back
        # non-finite probabilities, and json.dump writes a bare NaN token that json.load
        # reads back without complaint - so the bad value survives into the API, where
        # Starlette serializes with allow_nan=False and kills the whole /models response.
        # derma_v2 and derma_bin both shipped one for two days before it surfaced.
        "mean_confidence": (round(float(pt.max(1).mean()), 4)
                            if np.isfinite(pt).all() else None),
        "tta_used": use_tta, "tta_views": tta_name,
        "tta_decided_on": "val",
        "val_accuracy_no_tta": round(float(val_plain), 4),
        "val_accuracy_with_tta": round(float(val_tta), 4),
        # Both test numbers stay published. They are evidence, not a menu: the flag above
        # was already fixed by val before either of these was computed.
        "test_accuracy_no_tta": round(float(acc_plain), 4),
        "test_accuracy_with_tta": round(float(acc_tta), 4),
        "confusion_matrix": confusion_matrix(yt, pred).tolist(),
        "epoch_history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1), "device": DEVICE, "seed": SEED,
    }
    _primary_ckpt = {"state_dict": model.state_dict(), "size": SIZE, "classes": classes,
                # Without this, serving rebuilds a resnet18 and load_state_dict raises on
                # every key. Old checkpoints have no "arch" field, so the loader defaults to
                # resnet18 — which is what all twelve of them are.
                "arch": ARCH,
                # Serving MUST apply the same fixed preprocessing or it feeds the network
                # a different distribution than it was trained on - the exact failure mode
                # of the greyscale bug (step 57), just harder to notice.
                "preproc": PREPROC,
                # Training-time provenance. Not read at serve time - the augmentation only
                # ever runs during training - but without it there is no way to tell from a
                # served checkpoint whether it was trained under a policy that mirrors
                # anatomy. organc is served by a geometry_safe model precisely because that
                # matters, and "which recipe produced this file" should not require reading
                # a metrics JSON that can be moved or lost separately from the weights.
                "aug": AUG, "loss": LOSS, "selected_on": SELECT_ON,
                "dropout": DROPOUT, "mean": IM_MEAN, "std": IM_STD,
                "binary_task": BINARY, "tta": use_tta, "tta_views": tta_name,
                # serving side (quiz / case atlas) needs to rebuild the same label grouping
                # from the original MedMNIST labels — keep it with the weights, not in a
                # second copy that can drift.
                "medmnist": DATASET,
                "binary_positive": (BINARY_TASKS[DATASET]["positive"] if BINARY else None)}
    torch.save(_primary_ckpt, os.path.join(MODEL_DIR, KEY + ".pt"))
    # The alternate selections, written so verify_retrain_gains.py can score them on its own
    # CPU fp32 path. They carry the PRIMARY's tta flag on purpose: the TTA decision was made on
    # val for the primary, and re-deciding it per alternate would spend the same val split on a
    # second selection.
    metrics["selection_alternates"] = {}
    for _mn in sorted(alt_best):
        metrics["selection_alternates"][_mn] = {"val_score": round(alt_best[_mn]["score"], 4),
                                                "epoch": alt_best[_mn]["epoch"],
                                                "checkpoint": (KEY + ".pt") if _mn == SELECT_ON
                                                              else (KEY + "_sel" + _mn + ".pt")}
        if _mn == SELECT_ON:
            continue
        _ck = dict(_primary_ckpt)
        _ck["state_dict"], _ck["selected_on"] = alt_best[_mn]["state"], _mn
        torch.save(_ck, os.path.join(MODEL_DIR, KEY + "_sel" + _mn + ".pt"))
        print("  [select] alternate %-6s best val=%.4f at epoch %d -> %s_sel%s.pt"
              % (_mn, alt_best[_mn]["score"], alt_best[_mn]["epoch"], KEY, _mn), flush=True)
    json.dump(metrics, open(os.path.join(MODEL_DIR, KEY + "_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[report]\n", classification_report(yt, pred, target_names=classes, zero_division=0))
    print("[RESULT] %s test_acc=%.4f train_acc=%.4f gap=%+.4f macroF1=%s auc=%s tta=%s"
          % (KEY, test_acc, train_acc, train_acc - test_acc,
             metrics["test_macro_f1"], metrics["test_auc"], use_tta))
    print(KEY.upper() + "_DONE", flush=True)


if __name__ == "__main__":
    main()
