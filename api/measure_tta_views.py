# -*- coding: utf-8 -*-
"""
Score an already-trained checkpoint under every TTA view set, choose on val, report test.

TTA is a decision made after training. It costs inference only, so a wider view set can be
evaluated on models that are already on disk - no retraining, no GPU contention, nothing that
can be lost to a MemoryError. The Kaggle DR notebook in `code benifit/` averages five views
(identity, hflip, +/-10 degrees, centre-crop 0.9) where this project has only ever averaged two.

Rules this obeys, all of them already written down in TRAINING_LOG:

  - The view set is chosen on VAL and test is scored once (step 38: the TTA flag used to be
    picked by scoring test both ways, which reports max(a,b) of two test numbers).
  - Ties go to the cheapest set. Each view is a forward pass at serve time.
  - Mirror-containing sets are refused for datasets whose labels are not mirror-invariant
    (step 67: organc has kidney-left/kidney-right, and an hflip average asks the model to
    agree with its own mirror, which for those classes is the other class).
  - For binary tasks `disease_caught` is reported next to accuracy, and a set that catches
    less disease is not promoted however good its accuracy looks (step 14).

It reports. Writing the winning set into the checkpoint is `--write`, and is refused unless
val actually prefers it.

    python measure_tta_views.py                 # every served image model
    python measure_tta_views.py --write path    # apply the val-chosen set for one model
"""
import os, sys, json, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score, confusion_matrix

from nets import build_medmnist_backbone, MEDMNIST_ARCHS
import preproc

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
DEVICE = os.environ.get("TTAV_DEVICE", "cpu")
BATCH = int(os.environ.get("TTAV_BATCH", "16"))
VAL_MAX = int(os.environ.get("TTAV_VAL_MAX", "4000"))   # cap the val pass on the huge splits
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

SERVED = [("breast", "breastmnist"), ("derma", "dermamnist"), ("derma_bin", "dermamnist"),
          ("blood", "bloodmnist"), ("organc", "organcmnist"), ("path", "pathmnist"),
          ("oct", "octmnist"), ("oct_bin", "octmnist"),
          ("retina", "retinamnist"), ("retina_bin", "retinamnist")]

# Datasets whose labels change under a horizontal mirror. organc has kidney-left/right,
# lung-left/right, femur-left/right - 6 of 11 classes and 41.6% of its test set.
NOT_MIRROR_INVARIANT = {"organcmnist"}


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


def split_of(dataset, split, size, bp, pre, cap=0, seed=0):
    DataClass = getattr(medmnist, INFO[dataset]["python_class"])
    src = min([s for s in (64, 128, 224) if s >= size] or [224])
    ds = DataClass(split=split, download=True, size=src, root=DATA_ROOT)
    X, y = ds.imgs, ds.labels.astype(np.int64).reshape(-1)
    if bp:
        pos = set(bp)
        y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)
    if cap and len(y) > cap:
        # Stratified subsample, fixed seed. Only ever applied to VAL - test is never capped.
        rng = np.random.RandomState(seed)
        keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            keep += rng.choice(idx, min(len(idx), max(1, cap // len(np.unique(y)))),
                               replace=False).tolist()
        keep = np.sort(np.array(keep))
        X, y = X[keep], y[keep]
    if pre and pre != "none":
        X = np.stack([preproc.apply(X[i], pre) for i in range(len(X))])
    return X, y


@torch.no_grad()
def probs_for(net, X, y, size, views):
    out = []
    for xb, _ in DataLoader(DS(X, y, size), batch_size=BATCH, num_workers=0):
        out.append(preproc.tta_average(net, xb.to(DEVICE), views).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="store the val-chosen set in the checkpoint")
    ap.add_argument("keys", nargs="*")
    args = ap.parse_args()

    report = {}
    for key, dataset in SERVED:
        if args.keys and key not in args.keys:
            continue
        man = os.path.join(MODEL_DIR, f"{key}_ensemble.json")
        if os.path.exists(man):
            print("%-11s SKIP - served as an ensemble; each member keeps its own view set" % key)
            continue
        v2 = os.path.join(MODEL_DIR, f"{key}_v2.pt")
        cp = v2 if os.path.exists(v2) else os.path.join(MODEL_DIR, f"{key}.pt")
        if not os.path.exists(cp):
            print("%-11s SKIP - no checkpoint" % key)
            continue
        ck = torch.load(cp, map_location=DEVICE, weights_only=False)
        arch = ck.get("arch", "resnet18")
        if arch not in MEDMNIST_ARCHS:
            print("%-11s SKIP - arch %s" % (key, arch))
            continue
        size, classes = ck.get("size", 64), ck["classes"]
        net, _ = build_medmnist_backbone(arch, num_classes=len(classes), pretrained=False,
                                         dropout=ck.get("dropout", 0.0))
        net = net.to(DEVICE).eval()
        net.load_state_dict(ck["state_dict"])
        bp, pre = ck.get("binary_positive"), ck.get("preproc", "none")
        cur_views, cur_name = preproc.views_for(ck)

        cands = list(preproc.TTA_VIEWS)
        if dataset in NOT_MIRROR_INVARIANT:
            cands = [c for c in cands if "hflip" not in preproc.TTA_VIEWS[c]]

        Xv, yv = split_of(dataset, "val", size, bp, pre, cap=VAL_MAX)
        val = {c: accuracy_score(yv, probs_for(net, Xv, yv, size, preproc.TTA_VIEWS[c]).argmax(1))
               for c in cands}
        del Xv, yv
        # cheapest set wins ties - each view is a forward pass every time the model is served
        best = min(val, key=lambda c: (-val[c], len(preproc.TTA_VIEWS[c])))

        Xt, yt = split_of(dataset, "test", size, bp, pre)
        test, caught = {}, {}
        for c in cands:
            p = probs_for(net, Xt, yt, size, preproc.TTA_VIEWS[c])
            pred = p.argmax(1)
            test[c] = accuracy_score(yt, pred)
            if len(classes) == 2:
                pi = 1 if bp else (0 if dataset == "breastmnist" else 1)
                cm = confusion_matrix(yt, pred)
                caught[c] = (int(cm[pi][pi]), int(cm[pi].sum()))
        del Xt, yt, net

        print("\n%s  (arch=%s, currently '%s', n_val=%d capped, n_test=%d)"
              % (key, arch, cur_name, min(VAL_MAX, INFO[dataset]["n_samples"]["val"]),
                 INFO[dataset]["n_samples"]["test"]))
        for c in cands:
            mark = "  <- val pick" if c == best else ("  (current)" if c == cur_name else "")
            cc = ("  caught %d/%d" % caught[c]) if c in caught else ""
            print("   %-6s %d views  val=%.4f  test=%.4f%s%s"
                  % (c, len(preproc.TTA_VIEWS[c]), val[c], test[c], cc, mark))

        gain = test[best] - test.get(cur_name, test[best])
        verdict = "unchanged" if best == cur_name else "val prefers %s (test %+0.4f)" % (best, gain)
        # the step-14 guard, applied to the view set
        if best != cur_name and best in caught and cur_name in caught \
                and caught[best][0] < caught[cur_name][0]:
            verdict = ("REFUSED: %s catches %d/%d against the current %d/%d"
                       % (best, caught[best][0], caught[best][1],
                          caught[cur_name][0], caught[cur_name][1]))
            best = cur_name
        print("   -> %s" % verdict)

        report[key] = {"arch": arch, "current": cur_name, "val_pick": best,
                       "val": {c: round(float(v), 4) for c, v in val.items()},
                       "test": {c: round(float(v), 4) for c, v in test.items()},
                       "disease_caught": {c: "%d/%d" % v for c, v in caught.items()},
                       "verdict": verdict}

        if args.write and best != cur_name and "REFUSED" not in verdict:
            ck["tta_views"] = best
            ck["tta"] = best != "none"
            torch.save(ck, cp)
            mp = cp.replace(".pt", "_metrics.json")
            if os.path.exists(mp):
                with open(mp, encoding="utf-8") as f:
                    m = json.load(f)
                m.update({"tta_views": best, "tta_used": best != "none",
                          "tta_decided_on": "val", "test_accuracy": round(float(test[best]), 4)})
                with open(mp, "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=2)
            print("   [written] %s -> tta_views=%s" % (os.path.basename(cp), best))

    out = os.path.join(MODEL_DIR, "_tta_views.json")
    prev = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            prev = json.load(f)
    prev.update(report)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
