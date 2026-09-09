# -*- coding: utf-8 -*-
"""
Re-decide the TTA flag on VAL for every shipped MedMNIST checkpoint, and correct any
published number that only stood because the flag had been decided on TEST.

The bug (found 2026-09-07, session 6). train_medmnist_v2.py used to pick the TTA flag by
scoring the TEST split twice - once plain, once with hflip TTA - and keeping whichever won:

    acc_plain = accuracy_score(y_test, plain_predictions)
    acc_tta   = accuracy_score(y_test, tta_predictions)
    use_tta   = acc_tta >= acc_plain          # <-- both numbers are TEST numbers

That reports max(a, b) of two test-set measurements, so it can only ever move the headline
up. It is test-set selection, and the project's own rule (TRAINING_LOG step 21, rule 2) had
already forbidden it in writing: "test is not touched to select anything - even the TTA
decision is made on val". The trainer was not obeying its own rule.

Measured inflation in the shipped metrics files before this ran:
    retina_v2 +0.0275 (0.5800 -> 0.6075)    oct_v2   +0.0050
    oct_bin   +0.0040                        derma_v2 +0.0025
    breast_v2 / derma_bin / retina_bin       +0.0000

This script does the honest thing instead: decide on val, then report test under whatever
that decision was - including when the honest number is lower. A published number that drops
because it stopped being chosen on the test set was never ours to publish.

    python fix_tta_selection.py              # report only, changes nothing
    python fix_tta_selection.py --write      # rewrite ckpt["tta"] + the metrics file
"""
import os, json, argparse
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
DEVICE = os.environ.get("FIX_DEVICE", "cpu")   # cpu by default: the GPU may be training
# Activation memory scales with the batch, and this box has run as low as 0.44 GB free while
# a training job holds the GPU. FIX_BATCH=8 makes it safe to run alongside one.
BATCH = int(os.environ.get("FIX_BATCH", "32"))
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def discover_targets():
    """Every (key, checkpoint) whose metrics file still carries a test-selected TTA flag.

    Discovered rather than listed, because session 6 keeps adding experiment keys
    (retina_r50, retina_eb0, ...) and a hardcoded list would quietly skip exactly the runs
    that started before the trainer was fixed - the ones that most need correcting. Models
    already marked "tta_decided_on": "val" were produced by the fixed trainer and are left
    alone, so re-running this is idempotent.
    """
    import glob
    out = []
    for mp in sorted(glob.glob(os.path.join(MODEL_DIR, "*_metrics.json"))):
        key = os.path.basename(mp)[:-len("_metrics.json")]
        try:
            with open(mp, encoding="utf-8") as f:
                met = json.load(f)
        except Exception:
            continue
        if "tta_used" not in met or met.get("tta_decided_on") == "val":
            continue
        cp = os.path.join(MODEL_DIR, key + ".pt")
        if os.path.exists(cp):
            out.append((key, key + ".pt"))
    return out


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
def probs(net, X, y, size, tta, batch=None):
    """No DataLoader workers, and a batch small enough to sit next to a training job: this
    runs on an 8 GB box that has been measured at 0.44 GB free with the GPU busy. Session 2
    lost three runs to MemoryError doing less than this. Override with FIX_BATCH."""
    out = []
    for xb, _ in DataLoader(DS(X, y, size), batch_size=batch or BATCH, num_workers=0):
        xb = xb.to(DEVICE)
        p = torch.softmax(net(xb), 1)
        if tta:
            p = (p + torch.softmax(net(torch.flip(xb, dims=[3])), 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply the correction to disk")
    ap.add_argument("keys", nargs="*", help="subset of model keys (default: all)")
    args = ap.parse_args()
    if args.keys:
        # Naming a key explicitly overrides the "already corrected" filter. Needed because the
        # first pass rewrote test_accuracy from a CPU fp32 re-measure but left the fp16
        # test_accuracy_no_tta/with_tta fields from the original GPU run, so those files say
        # tta_used=False next to a no-TTA number that differs from the headline.
        todo = [(k, k + ".pt") for k in args.keys
                if os.path.exists(os.path.join(MODEL_DIR, k + ".pt"))]
    else:
        todo = discover_targets()
    if not todo:
        print("nothing to correct: every metrics file already says tta_decided_on=val")
        return

    report = {}
    print("%-12s %-24s %-24s %s" % ("model", "val  plain / tta", "test  plain / tta", "verdict"))
    print("-" * 100)
    for key, ckpt_name in todo:
        cp = os.path.join(MODEL_DIR, ckpt_name)
        mp = os.path.join(MODEL_DIR, key + "_metrics.json")
        if not (os.path.exists(cp) and os.path.exists(mp)):
            print("%-12s SKIP (missing checkpoint or metrics)" % key)
            continue
        with open(mp, encoding="utf-8") as f:
            met = json.load(f)
        ck = torch.load(cp, map_location=DEVICE, weights_only=False)
        size, classes = ck.get("size", 64), ck["classes"]
        arch = ck.get("arch", "resnet18")
        if arch not in MEDMNIST_ARCHS:
            # retina_ord is a CORAL head ("resnet18_coral"): a different output layer with a
            # different decision rule, so hflip-TTA is not the same knob there. It is also
            # not served (session 5). Skipping is correct; crashing on it is not.
            print("%-12s SKIP (arch=%s is not a plain classifier head)" % (key, arch))
            continue
        net, _ = build_medmnist_backbone(arch, num_classes=len(classes),
                                         pretrained=False, dropout=ck.get("dropout", 0.0))
        net = net.to(DEVICE).eval()
        net.load_state_dict(ck["state_dict"])
        bp = ck.get("binary_positive")

        pre = ck.get("preproc", "none")
        Xv, yv = split_of(ck["medmnist"], "val", size, bp, pre)
        vp = probs(net, Xv, yv, size, False)
        vt = probs(net, Xv, yv, size, True)
        val_plain = accuracy_score(yv, vp.argmax(1))
        val_tta = accuracy_score(yv, vt.argmax(1))
        use_tta = bool(val_tta > val_plain)         # ties keep the cheaper plain path
        del Xv, yv, vp, vt

        Xt, yt = split_of(ck["medmnist"], "test", size, bp, pre)
        tp = probs(net, Xt, yt, size, False)
        tt = probs(net, Xt, yt, size, True)
        acc_plain = accuracy_score(yt, tp.argmax(1))
        acc_tta = accuracy_score(yt, tt.argmax(1))
        pt = tt if use_tta else tp
        pred = pt.argmax(1)
        honest = accuracy_score(yt, pred)
        old_flag, old_acc = met.get("tta_used"), met.get("test_accuracy")
        delta = honest - old_acc

        verdict = ("unchanged" if abs(delta) < 1e-9 else
                   "CORRECTED %+0.4f  (%s -> %s)" % (delta, old_acc, round(float(honest), 4)))
        print("%-12s %8.4f / %8.4f       %8.4f / %8.4f       tta %s->%s  %s"
              % (key, val_plain, val_tta, acc_plain, acc_tta, old_flag, use_tta, verdict))

        report[key] = {
            "val_accuracy_no_tta": round(float(val_plain), 4),
            "val_accuracy_with_tta": round(float(val_tta), 4),
            "test_accuracy_no_tta": round(float(acc_plain), 4),
            "test_accuracy_with_tta": round(float(acc_tta), 4),
            "tta_used_before": bool(old_flag), "tta_used_after": use_tta,
            "test_accuracy_before": old_acc, "test_accuracy_after": round(float(honest), 4),
            "delta": round(float(delta), 4),
        }

        # Naming keys explicitly forces the write even when the headline does not move:
        # the point of a named re-run is to refresh the per-configuration fields, and
        # "the headline is already right" is not evidence that the rest of the file is.
        if args.write and (bool(args.keys) or use_tta != bool(old_flag) or abs(delta) > 1e-9):
            # A headline that came from a val-tuned decision threshold is NOT the argmax
            # number and must not be overwritten with it. oct_bin publishes 0.9920 at
            # threshold 0.440, chosen on val and cleared by the step-14 guard because it
            # caught MORE disease (742/750 vs 741/750), not less. Recomputing argmax there
            # reported 0.9910 and silently threw the tuning away - a correction that
            # under-reports is still a wrong number.
            tuned = "threshold" in str(met.get("test_accuracy_source", "")).lower()
            met.update({} if tuned else {
                "test_accuracy": round(float(honest), 4),
                # Re-measured here, not carried over: the originals came off the GPU under
                # fp16 and this pass runs CPU fp32, so leaving them would put a no-TTA number
                # next to a headline that disagrees with it while tta_used says False.
                "test_accuracy_no_tta": round(float(acc_plain), 4),
                "test_accuracy_with_tta": round(float(acc_tta), 4),
                "test_numbers_remeasured_on": DEVICE,
                "test_balanced_accuracy": round(float(balanced_accuracy_score(yt, pred)), 4),
                "test_macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
                "confusion_matrix": confusion_matrix(yt, pred).tolist(),
                "tta_used": use_tta, "tta_decided_on": "val",
                "val_accuracy_no_tta": round(float(val_plain), 4),
                "val_accuracy_with_tta": round(float(val_tta), 4),
                "tta_correction_2026_09_07": {
                    "was": old_acc, "now": round(float(honest), 4),
                    "why": ("the TTA flag had been selected on the test split, which reports "
                            "the max of two test numbers; it is now selected on val"),
                },
            })
            # The TTA metadata is corrected for a tuned model too - just not its headline.
            if tuned:
                met.update({
                    "tta_used": use_tta, "tta_decided_on": "val",
                    "val_accuracy_no_tta": round(float(val_plain), 4),
                    "val_accuracy_with_tta": round(float(val_tta), 4),
                    "test_accuracy_argmax_0.5": round(float(honest), 4),
                    "tta_correction_2026_09_07": {
                        "headline_unchanged": old_acc,
                        "why": ("val confirmed the same TTA decision, and this model's headline "
                                "comes from a val-tuned threshold, not from argmax - so there "
                                "was nothing to correct"),
                    },
                })
            try:
                ptn = pt / np.clip(pt.sum(1, keepdims=True), 1e-12, None)
                auc = (roc_auc_score(yt, ptn[:, 1]) if len(classes) == 2
                       else roc_auc_score(yt, ptn, multi_class="ovr", average="macro"))
                met["test_auc"] = round(float(auc), 4)
            except Exception as e:
                print("  [warn] AUC: %s: %s" % (type(e).__name__, e))
            with open(mp, "w", encoding="utf-8") as f:
                json.dump(met, f, ensure_ascii=False, indent=2)
            ck["tta"] = use_tta
            torch.save(ck, cp)
            print("      [written] %s + %s" % (os.path.basename(mp), ckpt_name))
        del net, Xt, yt, tp, tt, pt

    out = os.path.join(MODEL_DIR, "_tta_selection_fix.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
