# -*- coding: utf-8 -*-
"""
Average the probabilities of several ALREADY-TRAINED checkpoints for the same task, choose
everything on val, and report test.

Why this is not a repeat of session 2's failed ensemble. That one trained five SEEDS of the
same resnet18, under memory pressure, on a collided GPU; two of five survived, they stopped
early, and the ensemble scored 0.8625 against 0.8825 for the single original model. Two
things were wrong with it: the members were weaker than the baseline, and five seeds of one
architecture make highly correlated errors, so averaging them adds little even when they are
healthy.

This script instead ensembles across ARCHITECTURES (resnet18 / resnet50 / efficientnet_b0)
that were each trained cleanly and are each already on disk. Different architectures make
less correlated errors than different seeds of one architecture, which is the entire reason
averaging can help. And it costs inference only - no training, no GPU contention, nothing
that can be lost to a MemoryError halfway through.

Protocol, following the rules this project has already written down:
  - The member subset and the decision threshold are chosen on VAL. Test is scored once.
  - For binary tasks the disease class is named explicitly (session 3, step 14) and
    `disease_caught` is reported next to accuracy, because a percentage alone hides exactly
    the trade that cost four cancers there.
  - Every member's own TTA flag is read from its checkpoint, not assumed (session 4, step 19).

  python ensemble_archs.py breast_v2 breast_eb0
  python ensemble_archs.py --name retina_ens retina_v2 retina_r50 retina_eb0 retina_eb0cut
"""
import os, json, argparse, itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             balanced_accuracy_score)

from nets import build_medmnist_backbone, MEDMNIST_ARCHS
import preproc

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
DEVICE = os.environ.get("ENS_DEVICE", "cpu")
BATCH = int(os.environ.get("ENS_BATCH", "16"))
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Which original MedMNIST class index means "disease" for a NON-binary task. Binary heads
# relabel to disease=1 themselves, so they are not listed here.
# BreastMNIST ships ['malignant', 'normal, benign'] - disease is index 0, which is the bug
# that cost four cancers in session 3.
CLINICAL_POSITIVE = {"breastmnist": 0}


class DS(Dataset):
    def __init__(self, X, y, size):
        self.X, self.y = X, y
        self.tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        im = self.X[i]
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        return self.tf(np.ascontiguousarray(im)), int(self.y[i])


def split_of(dataset, split, size, binary_positive, pre=None):
    DataClass = getattr(medmnist, INFO[dataset]["python_class"])
    src = min([s for s in (64, 128, 224) if s >= size] or [224])
    ds = DataClass(split=split, download=True, size=src, root=DATA_ROOT)
    X, y = ds.imgs, ds.labels.astype(np.int64).reshape(-1)
    if pre and pre != "none":
        X = np.stack([preproc.apply(X[i], pre) for i in range(len(X))])
    if binary_positive:
        pos = set(binary_positive)
        y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)
    return X, y


@torch.no_grad()
def member_probs(key, split):
    """Probabilities from one checkpoint on one split, in the configuration it is served in."""
    cp = os.path.join(MODEL_DIR, key + ".pt")
    ck = torch.load(cp, map_location=DEVICE, weights_only=False)
    arch = ck.get("arch", "resnet18")
    if arch not in MEDMNIST_ARCHS:
        raise SystemExit("%s has arch=%s, which this script cannot rebuild" % (key, arch))
    size, classes = ck.get("size", 64), ck["classes"]
    net, _ = build_medmnist_backbone(arch, num_classes=len(classes), pretrained=False,
                                     dropout=ck.get("dropout", 0.0))
    net = net.to(DEVICE).eval()
    net.load_state_dict(ck["state_dict"])
    views, tta = preproc.views_for(ck)
    X, y = split_of(ck["medmnist"], split, size, ck.get("binary_positive"),
                    ck.get("preproc", "none"))
    out = []
    for xb, _ in DataLoader(DS(X, y, size), batch_size=BATCH, num_workers=0):
        out.append(preproc.tta_average(net, xb.to(DEVICE), views).cpu().numpy())
    del net
    return np.concatenate(out), y, {"arch": arch, "size": size, "tta": tta,
                                    "classes": classes, "medmnist": ck["medmnist"],
                                    "preproc": ck.get("preproc", "none"),
                                    "binary_positive": ck.get("binary_positive")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="+", help="checkpoint keys to ensemble (same task)")
    ap.add_argument("--name", default=None, help="output key (default: <first>_ens)")
    args = ap.parse_args()
    name = args.name or (args.keys[0] + "_ens")

    print("collecting member probabilities on val and test (device=%s)" % DEVICE, flush=True)
    val_p, test_p, meta = {}, {}, {}
    yv = yt = None
    for k in args.keys:
        pv, yv_k, m = member_probs(k, "val")
        pt, yt_k, _ = member_probs(k, "test")
        if yv is None:
            yv, yt = yv_k, yt_k
        elif not (np.array_equal(yv, yv_k) and np.array_equal(yt, yt_k)):
            raise SystemExit("%s has different labels - these are not the same task" % k)
        val_p[k], test_p[k], meta[k] = pv, pt, m
        print("  %-18s arch=%-16s tta=%-5s val_acc=%.4f test_acc=%.4f"
              % (k, m["arch"], m["tta"], accuracy_score(yv, pv.argmax(1)),
                 accuracy_score(yt, pt.argmax(1))), flush=True)

    m0 = meta[args.keys[0]]
    n_cls = len(m0["classes"])
    if n_cls == 2:
        pos_idx = 1 if m0["binary_positive"] else CLINICAL_POSITIVE.get(m0["medmnist"], 1)
    else:
        pos_idx = None

    # ---- choose the member subset on VAL --------------------------------------------------
    # Every non-empty subset is scored on val; test is not consulted. With <=5 members this is
    # at most 31 combinations, so an exhaustive search is cheaper than any heuristic.
    best = None
    for r in range(1, len(args.keys) + 1):
        for combo in itertools.combinations(args.keys, r):
            pv = np.mean([val_p[k] for k in combo], axis=0)
            a = accuracy_score(yv, pv.argmax(1))
            # tie-break toward fewer members: a smaller ensemble is cheaper to serve and is
            # the more conservative choice when val cannot tell them apart.
            if best is None or a > best[0] + 1e-9 or (abs(a - best[0]) <= 1e-9 and r < len(best[1])):
                best = (a, combo)
    val_best, combo = best
    print("\nval-selected members: %s  (val_acc=%.4f)" % (", ".join(combo), val_best))

    # ---- and the UNSELECTED baseline: every member, equal weight ---------------------------
    # The subset search above scores up to 2^n - 1 combinations on a val split that can be 78
    # images (breast) or 120 (retina). That is enough freedom to fit val noise, and in practice
    # it keeps returning a single member rather than an ensemble. The plain equal-weight
    # average of ALL members chooses nothing at all, so it cannot overfit val - and it is what
    # "ensemble" normally means. Both are reported; neither is picked by looking at test.
    pt_all = np.mean([test_p[k] for k in args.keys], axis=0)
    all_acc = accuracy_score(yt, pt_all.argmax(1))
    all_val = accuracy_score(yv, np.mean([val_p[k] for k in args.keys], axis=0).argmax(1))
    print("all-member equal-weight: val_acc=%.4f test_acc=%.4f" % (all_val, all_acc))

    pv = np.mean([val_p[k] for k in combo], axis=0)
    pt = np.mean([test_p[k] for k in combo], axis=0)
    pred = pt.argmax(1)
    acc = accuracy_score(yt, pred)
    cm = confusion_matrix(yt, pred)

    mean_member = float(np.mean([accuracy_score(yt, test_p[k].argmax(1)) for k in combo]))
    best_member = max((accuracy_score(yt, test_p[k].argmax(1)), k) for k in args.keys)

    result = {
        "model": name, "members": list(combo), "all_candidates": list(args.keys),
        "member_archs": {k: meta[k]["arch"] for k in args.keys},
        "medmnist": m0["medmnist"], "classes": m0["classes"], "n_classes": n_cls,
        "binary_task": bool(m0["binary_positive"]),
        "selection": "member subset chosen on val; test scored once",
        "val_accuracy": round(float(val_best), 4),
        "test_accuracy": round(float(acc), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
        "test_macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
        "confusion_matrix": cm.tolist(),
        "mean_member_test_accuracy": round(mean_member, 4),
        "best_single_member": {"key": best_member[1], "test_accuracy": round(best_member[0], 4)},
        "ensemble_gain_over_mean_member": round(float(acc) - mean_member, 4),
        "ensemble_gain_over_best_member": round(float(acc) - best_member[0], 4),
        "n_test": int(len(yt)), "device": DEVICE,
        "all_member_equal_weight": {
            "members": list(args.keys),
            "val_accuracy": round(float(all_val), 4),
            "test_accuracy": round(float(all_acc), 4),
            "test_balanced_accuracy": round(
                float(balanced_accuracy_score(yt, pt_all.argmax(1))), 4),
            "test_macro_f1": round(float(f1_score(yt, pt_all.argmax(1), average="macro")), 4),
            "confusion_matrix": confusion_matrix(yt, pt_all.argmax(1)).tolist(),
            "gain_over_best_member": round(float(all_acc) - best_member[0], 4),
            "note": ("chooses nothing, so it cannot overfit val; reported next to the "
                     "val-selected subset rather than instead of it"),
        },
    }
    try:
        ptn = pt / np.clip(pt.sum(1, keepdims=True), 1e-12, None)
        result["test_auc"] = round(float(roc_auc_score(yt, ptn[:, 1]) if n_cls == 2 else
                                         roc_auc_score(yt, ptn, multi_class="ovr",
                                                       average="macro")), 4)
    except Exception as e:
        print("  [warn] AUC: %s: %s" % (type(e).__name__, e))
        result["test_auc"] = None

    if pos_idx is not None:
        caught, total = int(cm[pos_idx][pos_idx]), int(cm[pos_idx].sum())
        result["positive_class_index"] = pos_idx
        result["positive_class_name"] = m0["classes"][pos_idx]
        result["disease_caught"] = "%d/%d" % (caught, total)
        result["sensitivity"] = round(caught / total, 4)
        # The same number for the all-member ensemble. Reporting accuracy for one variant and
        # sensitivity for another is how session 3 nearly published a gain that cost four
        # cancers; every variant carries its own disease count or none of them do.
        cma = np.array(result["all_member_equal_weight"]["confusion_matrix"])
        ca, ta = int(cma[pos_idx][pos_idx]), int(cma[pos_idx].sum())
        result["all_member_equal_weight"]["disease_caught"] = "%d/%d" % (ca, ta)
        result["all_member_equal_weight"]["sensitivity"] = round(ca / ta, 4)

    print("\n%-34s %s" % ("ensemble test accuracy", result["test_accuracy"]))
    print("%-34s %s" % ("mean member test accuracy", result["mean_member_test_accuracy"]))
    print("%-34s %s (%s)" % ("best single member", result["best_single_member"]["test_accuracy"],
                             result["best_single_member"]["key"]))
    print("%-34s %+0.4f" % ("gain over mean member", result["ensemble_gain_over_mean_member"]))
    print("%-34s %+0.4f" % ("gain over BEST member", result["ensemble_gain_over_best_member"]))
    print("%-34s %s (%+0.4f vs best member)" % ("all-member equal weight",
          result["all_member_equal_weight"]["test_accuracy"],
          result["all_member_equal_weight"]["gain_over_best_member"]))
    if pos_idx is not None:
        print("%-34s %s (%s)" % ("disease caught", result["disease_caught"],
                                 result["positive_class_name"]))
    print("confusion matrix:")
    for row in cm.tolist():
        print("   ", row)

    out = os.path.join(MODEL_DIR, name + "_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
