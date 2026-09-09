#!/usr/bin/env python
"""Score every per-metric checkpoint of a run and print what each selection actually bought.

Usage:  python compare_selections.py derma_bin_all3 derma_bin_focal3

The trainer keeps one checkpoint per selection metric (acc / bacc / recall) off a SINGLE
trajectory, because separate runs cannot answer "which epoch should we keep" - this trainer does
not reproduce itself once the backbone unfreezes, by up to 0.0220 val accuracy (TRAINING_LOG
steps 91, 94). Reading three checkpoints out of one run removes that term entirely.

Accuracy alone cannot rank these: moving the selection toward the rare class trades false alarms
for caught disease by construction. So this prints the confusion matrix and the caught/total for
the disease class next to the accuracy, which is the trade the choice is actually about
(the promotion guard of step 62).

Measured on the independent CPU fp32 path in verify_retrain_gains.evaluate - not the training
process's own numbers.
"""
import json, os, sys
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_retrain_gains import MODEL_DIR, evaluate   # noqa: E402
import torch                                            # noqa: E402


def disease_index(ck):
    """Which column is the disease. Mirrors the trainer and the threshold tuner.

    breastmnist keeps MedMNIST's original order where malignant is index 0; every relabelled
    binary head puts disease at index 1. Getting this backwards is the class-polarity bug of
    session 3 step 14, which tuned a threshold on the healthy class for weeks.
    """
    if ck.get("medmnist") == "breastmnist" and not ck.get("binary_task"):
        return 0
    return 1


def main():
    keys = sys.argv[1:]
    if not keys:
        raise SystemExit(__doc__)
    out = {}
    for key in keys:
        mp = os.path.join(MODEL_DIR, key + "_metrics.json")
        if not os.path.exists(mp):
            print("%s: no metrics file - run it first" % key)
            continue
        m = json.load(open(mp, encoding="utf-8"))
        alts = m.get("selection_alternates")
        if not alts:
            print("%s: no selection_alternates - trained before the per-metric change" % key)
            continue
        print("\n=== %s  (loss=%s, %d epochs, %s) ===" % (key, m.get("loss"), len(m["epoch_history"]),
                                                          m.get("medmnist")))
        print("%-8s %-6s %-9s %-8s %-8s %-8s %-8s %s"
              % ("select", "epoch", "val_score", "test", "bal_acc", "macroF1", "auc", "disease / false alarms"))
        rows = {}
        for metric in ("acc", "bacc", "recall"):
            if metric not in alts:
                continue
            ck_name = alts[metric]["checkpoint"][:-3]
            cp = os.path.join(MODEL_DIR, ck_name + ".pt")
            if not os.path.exists(cp):
                print("%-8s checkpoint missing: %s" % (metric, cp))
                continue
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            acc, size, n, src, auc, y, p = evaluate(cp, ck["medmnist"], ck.get("tta", False))
            pred = p.argmax(1)
            cm = confusion_matrix(y, pred)
            di = disease_index(ck)
            caught, total = int(cm[di, di]), int(cm[di].sum())
            false_alarms = int(cm.sum(0)[di] - cm[di, di])
            print("%-8s %-6s %-9.4f %-8.4f %-8.4f %-8.4f %-8s %d/%d  (+%d)"
                  % (metric, alts[metric]["epoch"], alts[metric]["val_score"], acc,
                     balanced_accuracy_score(y, pred), f1_score(y, pred, average="macro"),
                     "-" if auc is None else "%.4f" % auc, caught, total, false_alarms))
            rows[metric] = {"epoch": alts[metric]["epoch"], "val_score": alts[metric]["val_score"],
                            "test_accuracy": round(acc, 4),
                            "test_balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
                            "test_macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
                            "test_auc": None if auc is None else round(float(auc), 4),
                            "confusion_matrix": cm.tolist(),
                            "disease_caught": "%d/%d" % (caught, total), "false_alarms": false_alarms,
                            "checkpoint": ck_name + ".pt"}
        out[key] = {"loss": m.get("loss"), "epochs": len(m["epoch_history"]),
                    "n_test": n, "selections": rows}
    if out:
        pth = os.path.join(MODEL_DIR, "_selection_comparison.json")
        prev = json.load(open(pth, encoding="utf-8")) if os.path.exists(pth) else {}
        prev.update(out)
        json.dump(prev, open(pth, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("\nwrote models/_selection_comparison.json")


if __name__ == "__main__":
    main()
