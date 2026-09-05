# -*- coding: utf-8 -*-
"""
Post-hoc decision-threshold tuning for the binary models — no retraining.

A 2-class network is served with argmax, i.e. a fixed 0.5 cut on the positive probability.
0.5 is a default, not a result: it is optimal only when the classes are balanced and the
costs are symmetric, and neither holds here (derma_bin is 80/20 benign, and missing a
melanoma is not equal to a false alarm). Moving the cut costs one inference pass over the
validation split and can be worth a point or two of accuracy — versus hours for a retrain.

Two operating points are produced per model, both chosen on VALIDATION and then measured
once on the untouched test split:

  accuracy  — the threshold maximising validation accuracy. Comparable to benchmarks.
  screening — the highest threshold still reaching >=90% sensitivity on validation. This is
              the one a triage tool should run: it deliberately trades specificity away to
              stop missing positives.

The tuned thresholds are written into the checkpoint and the metrics JSON. Nothing here ever
looks at the test split to make a choice — test is only ever measured.

  python tune_threshold.py                     # every binary model found
  python tune_threshold.py derma_bin           # one
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

from nets import build_brain_resnet

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
DEVICE = os.environ.get("TUNE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
MIN_SENS = float(os.environ.get("MIN_SENS", "0.90"))

CANDIDATES = ["breast_v2", "derma_bin", "retina_bin", "oct_bin"]

# WHICH CLASS INDEX IS THE DISEASE. Do not assume 1. The `_bin` heads remap labels onto
# {0,1} with disease=1, so 1 is correct for them. BreastMNIST is NOT: its native order is
# ['malignant', 'normal, benign'], so disease sits at index 0. Assuming 1 does not raise an
# error - it silently tunes the threshold to detect HEALTH. Measured on breast_v2: that
# "improvement" raised accuracy 0.8782 -> 0.8910 while catching 4 FEWER cancers (33 -> 29).
CLINICAL_POSITIVE = {"breast_v2": 0, "breast": 0}     # default 1 for everything else


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


@torch.no_grad()
def probs(net, X, y, size, tta):
    out = []
    for xb, _ in DataLoader(DS(X, y, size), batch_size=64):
        xb = xb.to(DEVICE)
        p = torch.softmax(net(xb), 1)
        if tta:
            p = (p + torch.softmax(net(torch.flip(xb, dims=[3])), 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def split_of(dataset, split, size, binary_positive):
    DataClass = getattr(medmnist, INFO[dataset]["python_class"])
    ds = DataClass(split=split, download=True, size=size, root=DATA_ROOT)
    y = ds.labels.astype(np.int64).reshape(-1)
    if binary_positive:
        pos = set(binary_positive)
        y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)
    return ds.imgs, y


def rates(y, pred, pos):
    """Rates with respect to the DISEASE class `pos`, not blindly class 1."""
    yb, pb = (y == pos).astype(int), (pred == pos).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "positive_class_index": int(pos),
        "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
        "specificity": round(float(tn / max(tn + fp, 1)), 4),
        "disease_caught": "%d/%d" % (int(tp), int(tp + fn)),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "confusion_matrix_[[tn,fp],[fn,tp]]": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def tune(key):
    ckpt_path = os.path.join(MODEL_DIR, key + ".pt")
    if not os.path.exists(ckpt_path):
        print("%-12s no checkpoint" % key); return None
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if len(ck["classes"]) != 2:
        print("%-12s not binary, skipping" % key); return None
    dataset = ck.get("medmnist")
    if not dataset:
        print("%-12s checkpoint has no 'medmnist' field, skipping" % key); return None
    size, tta = ck.get("size", 224), bool(ck.get("tta", False))
    bp = ck.get("binary_positive")

    Xva, yva = split_of(dataset, "val", size, bp)
    Xte, yte = split_of(dataset, "test", size, bp)
    net = build_brain_resnet(num_classes=2, pretrained=False,
                             dropout=ck.get("dropout", 0.0)).to(DEVICE).eval()
    net.load_state_dict(ck["state_dict"])
    pv, pt = probs(net, Xva, yva, size, tta), probs(net, Xte, yte, size, tta)
    del net
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    pos = CLINICAL_POSITIVE.get(key, 1)
    neg = 1 - pos
    print("   disease class = index %d (%s)" % (pos, ck["classes"][pos]))
    sv, st = pv[:, pos], pt[:, pos]                    # P(disease) on val and on test
    pred_at = lambda score, t: np.where(score >= t, pos, neg)

    grid = np.arange(0.02, 0.99, 0.005)
    val_acc = np.array([accuracy_score(yva, pred_at(sv, t)) for t in grid])
    t_acc = float(grid[int(np.argmax(val_acc))])

    val_sens = np.array([((sv >= t)[yva == pos]).mean() for t in grid])
    ok = grid[val_sens >= MIN_SENS]
    t_scr = float(ok.max()) if len(ok) else float(grid[0])

    base = rates(yte, pt.argmax(1), pos)
    acc_pt = rates(yte, pred_at(st, t_acc), pos)
    scr_pt = rates(yte, pred_at(st, t_scr), pos)
    try:
        auc = float(roc_auc_score((yte == pos).astype(int), st))
    except Exception as e:
        print("  [warn] AUC failed: %s: %s" % (type(e).__name__, e)); auc = None

    print("%-12s n_val=%d n_test=%d  AUC=%s" % (key, len(yva), len(yte),
                                                "%.4f" % auc if auc else "-"))
    for name, t, r in (("default 0.50", 0.5, base),
                       ("accuracy  %.3f" % t_acc, t_acc, acc_pt),
                       ("screening %.3f" % t_scr, t_scr, scr_pt)):
        print("   %-18s acc=%.4f  sens=%.4f  spec=%.4f  macroF1=%.4f  caught %s"
              % (name, r["accuracy"], r["sensitivity"], r["specificity"], r["macro_f1"],
                 r["disease_caught"]))

    ck["threshold"] = t_acc
    ck["threshold_screening"] = t_scr
    torch.save(ck, ckpt_path)

    mp = os.path.join(MODEL_DIR, key + "_metrics.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            mj = json.load(f)
        if mj.get("test_auc") is None and auc is not None:
            mj["test_auc"] = round(auc, 4)
            mj["test_auc_source"] = "computed by tune_threshold.py (training run recorded none)"
        mj["threshold_tuning"] = {
            "method": ("thresholds chosen on the validation split only, then measured once on "
                       "the untouched test split; no retraining"),
            "default_0.5": base,
            "accuracy_optimal": {"threshold": round(t_acc, 3), **acc_pt},
            "screening_min_sensitivity_%.2f" % MIN_SENS: {"threshold": round(t_scr, 3), **scr_pt},
            "note": ("Serve the screening point for triage and quote the accuracy point for "
                     "benchmark comparison. Missing a positive is not equal to a false alarm."),
        }
        # Promote the tuned point to the headline ONLY if it does not detect fewer diseased
        # cases than the served argmax. Measured on breast_v2: the accuracy-optimal cut
        # raised accuracy 0.8782 -> 0.8910 while dropping from 33/42 to 29/42 cancers caught.
        # A higher accuracy bought with missed cancers is not an improvement, and promoting
        # it silently is how a metrics file starts lying.
        if acc_pt["accuracy"] > mj.get("test_accuracy", 0):
            if acc_pt["sensitivity"] >= base["sensitivity"]:
                mj["test_accuracy_argmax_0.5"] = mj.get("test_accuracy")
                mj["test_accuracy"] = acc_pt["accuracy"]
                mj["test_accuracy_source"] = "val-tuned decision threshold %.3f" % t_acc
            else:
                mj["headline_not_promoted"] = (
                    "The val-tuned threshold %.3f scores %.4f (vs %.4f) but catches only %s "
                    "of the disease against %s at argmax, so the headline was left alone."
                    % (t_acc, acc_pt["accuracy"], mj.get("test_accuracy"),
                       acc_pt["disease_caught"], base["disease_caught"]))
                print("   [kept] headline unchanged: tuned point catches %s vs %s"
                      % (acc_pt["disease_caught"], base["disease_caught"]))
        with open(mp, "w", encoding="utf-8") as f:
            json.dump(mj, f, ensure_ascii=False, indent=2)
    return {"key": key, "t_acc": t_acc, "t_scr": t_scr,
            "default": base, "accuracy": acc_pt, "screening": scr_pt, "auc": auc}


def main():
    keys = sys.argv[1:] or CANDIDATES
    print("device=%s  min_sensitivity=%.2f\n" % (DEVICE, MIN_SENS))
    out = [r for r in (tune(k) for k in keys) if r]
    with open(os.path.join(MODEL_DIR, "_threshold_tuning.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote models/_threshold_tuning.json")


if __name__ == "__main__":
    main()
