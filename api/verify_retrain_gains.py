# -*- coding: utf-8 -*-
"""
Integrity check on the 2026-09-04 retrain: re-measure the OLD and NEW checkpoints side by
side, in one process, through one evaluation path.

Why bother. retina went 0.495 -> 0.6075, which is above the published MedMNIST v2 ResNet-18
baseline (~0.51). A jump that size has two explanations: the recipe really is better, or the
new evaluation path is measuring something easier than the old one did. Those look identical
from a metrics file. They stop looking identical the moment you run BOTH checkpoints through
the SAME code on the SAME split:

  - if v1 reproduces the ~0.495 recorded back in July, the harness is sound and the v2 gain
    is real;
  - if v1 suddenly scores much higher too, the gain belongs to the evaluation, not the model,
    and the new numbers must not be published.

Also re-checks that the test split really is the official held-out one and reports how many
test images appear in train (should be zero — MedMNIST splits are fixed and disjoint, but
the brain-MRI leak in this same repo is why we no longer take that on faith).

  python verify_retrain_gains.py                 # all pairs found
  python verify_retrain_gains.py retina          # one
"""
import os, sys, json, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score, roc_auc_score

from nets import build_brain_resnet

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
DEVICE = os.environ.get("VERIFY_DEVICE", "cpu")   # cpu by default: the GPU may be training
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# (key, old checkpoint, new checkpoint, medmnist dataset, recorded old acc, recorded new acc)
PAIRS = [
    ("breast", "breast.pt", "breast_v2.pt", "breastmnist"),
    ("retina", "retina.pt", "retina_v2.pt", "retinamnist"),
    ("derma",  "derma.pt",  "derma_v2.pt",  "dermamnist"),
    ("oct",    "oct.pt",    "oct_v2.pt",    "octmnist"),
]


class DS(Dataset):
    def __init__(self, X, y, size):
        self.X, self.y = X, y
        self.tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        im = self.X[i]
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        return self.tf(np.ascontiguousarray(im)), int(self.y[i])


def recorded(name):
    p = os.path.join(MODEL_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f).get("test_accuracy")


@torch.no_grad()
def evaluate(ckpt_path, dataset, tta):
    """Load a checkpoint, evaluate on the official test split at the checkpoint's OWN size.

    TTA is read from the CHECKPOINT, not assumed. The v2 recipe evaluates hflip-TTA and keeps
    it only when it actually helped, so "v2" does not imply "TTA on": breast_v2 and derma_bin
    are both served with it off. Forcing it on here measured breast_v2 at 0.8654 instead of
    its served 0.8782 - exactly its recorded `test_accuracy_with_tta`. A verification harness
    that evaluates a model in a configuration nobody serves is verifying the wrong thing.
    """
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    tta = bool(ck.get("tta", tta))
    size = ck.get("size", 64)
    classes = ck["classes"]
    DataClass = getattr(medmnist, INFO[dataset]["python_class"])
    # Load the split at the resolution the checkpoint expects; MedMNIST caches per size.
    src = min([s for s in (64, 128, 224) if s >= size] or [224])
    ds = DataClass(split="test", download=True, size=src, root=DATA_ROOT)
    X, y = ds.imgs, ds.labels.astype(np.int64).reshape(-1)
    if len(classes) == 2 and ck.get("binary_positive"):
        pos = set(ck["binary_positive"])
        y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)

    net = build_brain_resnet(num_classes=len(classes), pretrained=False,
                             dropout=ck.get("dropout", 0.0)).to(DEVICE).eval()
    net.load_state_dict(ck["state_dict"])
    ps = []
    for xb, _ in DataLoader(DS(X, y, size), batch_size=64):
        xb = xb.to(DEVICE)
        p = torch.softmax(net(xb), 1)
        if tta:
            p = (p + torch.softmax(net(torch.flip(xb, dims=[3])), 1)) / 2
        ps.append(p.cpu().numpy())
    del net
    p = np.concatenate(ps)
    # Renormalize rows before AUC: fp16 softmax + flip-averaging leaves them at ~0.9995 and
    # sklearn rejects that as "not probabilities". Rescaling preserves every ranking.
    try:
        pn = p / np.clip(p.sum(1, keepdims=True), 1e-12, None)
        auc = (roc_auc_score(y, pn[:, 1]) if len(classes) == 2
               else roc_auc_score(y, pn, multi_class="ovr", average="macro"))
    except Exception as e:
        print("    [warn] AUC failed: %s: %s" % (type(e).__name__, e))
        auc = None
    return float(accuracy_score(y, p.argmax(1))), size, len(y), src, auc


def leak_check(dataset, size):
    """Exact-duplicate check between the official train and test splits (md5 of raw bytes)."""
    DataClass = getattr(medmnist, INFO[dataset]["python_class"])
    tr = DataClass(split="train", download=True, size=size, root=DATA_ROOT).imgs
    te = DataClass(split="test", download=True, size=size, root=DATA_ROOT).imgs
    h_tr = {hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest() for a in tr}
    dup = sum(1 for a in te if hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest() in h_tr)
    return dup, len(te)


def main():
    only = sys.argv[1:]
    print("device=%s\n" % DEVICE)
    print("%-8s %-22s %8s %8s   %s" % ("model", "checkpoint", "measured", "recorded", "verdict"))
    print("-" * 92)
    out = {}
    for key, old, new, dataset in PAIRS:
        if only and key not in only:
            continue
        row = {}
        for tag, ck_name, metrics_name, tta in (
                ("v1", old, "%s_metrics.json" % key, False),
                ("v2", new, "%s_v2_metrics.json" % key, True)):
            path = os.path.join(MODEL_DIR, ck_name)
            if not os.path.exists(path):
                print("%-8s %-22s %8s %8s   checkpoint not present" % (key, ck_name, "-", "-"))
                continue
            acc, size, n, src, auc = evaluate(path, dataset, tta)
            rec = recorded(metrics_name)
            if rec is None:
                verdict = "no recorded value to compare"
            elif abs(acc - rec) <= 0.02:
                verdict = "REPRODUCES (delta %+.4f)" % (acc - rec)
            else:
                verdict = "MISMATCH (delta %+.4f) <-- investigate" % (acc - rec)
            print("%-8s %-22s %7.2f%% %7s   %s  [%dpx from %dpx source, n=%d]"
                  % (key, ck_name, acc * 100,
                     ("%.2f%%" % (rec * 100)) if rec else "-", verdict, size, src, n))
            row[tag] = {"measured": round(acc, 4), "recorded": rec, "size": size,
                        "n_test": n, "auc": None if auc is None else round(float(auc), 4)}
            # Backfill an AUC the training run failed to record (derma_v2 hit the sklearn
            # sum-to-1 check before that was fixed). Only ever fills a null, never overwrites.
            mp = os.path.join(MODEL_DIR, metrics_name)
            if auc is not None and os.path.exists(mp):
                with open(mp, encoding="utf-8") as f:
                    mj = json.load(f)
                if mj.get("test_auc") is None:
                    mj["test_auc"] = round(float(auc), 4)
                    mj["test_auc_source"] = ("backfilled by verify_retrain_gains.py — the "
                                             "training run's AUC failed on the fp16 sum-to-1 check")
                    with open(mp, "w", encoding="utf-8") as f:
                        json.dump(mj, f, ensure_ascii=False, indent=2)
                    print("           -> backfilled test_auc=%.4f into %s" % (auc, metrics_name))
        if "v1" in row and "v2" in row:
            gain = row["v2"]["measured"] - row["v1"]["measured"]
            print("%-8s %-22s %8s %8s   same-harness gain: %+.4f" % ("", "-> v2 vs v1", "", "", gain))
            row["same_harness_gain"] = round(gain, 4)
        out[key] = row
        print()

    print("exact train/test duplicate check (md5 of raw pixels)")
    print("-" * 92)
    for key, _, _, dataset in PAIRS:
        if only and key not in only:
            continue
        try:
            dup, n = leak_check(dataset, 64)
            print("%-8s %d/%d test images have a byte-identical twin in train %s"
                  % (key, dup, n, "" if dup == 0 else "<-- LEAK"))
            out.setdefault(key, {})["exact_train_test_duplicates"] = dup
        except Exception as e:
            print("%-8s leak check failed: %s" % (key, e))

    with open(os.path.join(MODEL_DIR, "_verify_retrain.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote models/_verify_retrain.json")


if __name__ == "__main__":
    main()
