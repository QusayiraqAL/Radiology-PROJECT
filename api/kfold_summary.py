# -*- coding: utf-8 -*-
"""
Pool the K-fold runs of one or more configurations and say which configuration val actually
prefers - on a selection signal big enough to mean something.

The problem this solves is documented in TRAINING_LOG step 45. breast has 78 validation
images. Three candidates there ranked 0.9615 / 0.9487 / 0.9359 on val - 75, 74 and 73 correct,
one image apart each - and on test they ranked in exactly the REVERSE order. A difference of
one or two images is a coin flip, so no promotion on breast was defensible, and
`promote_model.py` correctly refuses it to this day.

K-fold does not create information, but it uses all of it. Pooling train+val gives 624 labelled
non-test images for breast and 1200 for retina; rotating the held-out fold means every one of
them is predicted exactly once by a model that never saw it. The comparison is then over 624
held-out predictions instead of 78 - and, just as importantly, it comes with a per-fold spread,
so "config A beat config B" can be checked against how much the folds themselves disagree.

The official test split is never involved. This ranks configurations; it does not measure them.

    python kfold_summary.py breast_k_r18 breast_k_eb0
    python kfold_summary.py --k 5 breast_k_r18 breast_k_eb0
"""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")


def fold_metrics(prefix, k):
    """Best val accuracy of each fold run named <prefix>f<i>."""
    out = []
    for i in range(k):
        p = os.path.join(MODEL_DIR, "%sf%d_metrics.json" % (prefix, i))
        if not os.path.exists(p):
            out.append(None)
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        h = m.get("epoch_history") or []
        out.append({
            "best_val": max((e["val_acc"] for e in h), default=None),
            "n_val": m.get("n_val"), "n_train": m.get("n_train"),
            "arch": m.get("arch", "resnet18"), "preproc": m.get("preproc", "none"),
            # test is recorded because the training script always measures it, but it is NOT
            # what this tool compares on - that would be the very thing step 38 removed.
            "test_accuracy": m.get("test_accuracy"),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("prefixes", nargs="+", help="run-name prefixes, e.g. breast_k_r18")
    args = ap.parse_args()

    print("%-18s %-16s %-11s %7s %9s %9s   %s"
          % ("config", "arch", "preproc", "folds", "mean val", "std", "per-fold"))
    print("-" * 104)
    table = {}
    for pre in args.prefixes:
        rows = fold_metrics(pre, args.k)
        got = [r for r in rows if r and r["best_val"] is not None]
        if not got:
            print("%-18s no fold results found" % pre)
            continue
        vals = np.array([r["best_val"] for r in got], dtype=float)
        arch = got[0]["arch"]
        pp = got[0]["preproc"]
        print("%-18s %-16s %-11s %3d/%-3d %9.4f %9.4f   %s"
              % (pre, arch, pp, len(got), args.k, vals.mean(), vals.std(),
                 " ".join("%.4f" % v for v in vals)))
        table[pre] = {"arch": arch, "preproc": pp, "folds_done": len(got), "k": args.k,
                      "mean_val": round(float(vals.mean()), 4),
                      "std_val": round(float(vals.std()), 4),
                      "per_fold_val": [round(float(v), 4) for v in vals],
                      "pooled_val_images": int(sum(r["n_val"] or 0 for r in got)),
                      "per_fold_test": [r["test_accuracy"] for r in got]}

    if len(table) >= 2:
        ranked = sorted(table.items(), key=lambda kv: -kv[1]["mean_val"])
        best, second = ranked[0], ranked[1]
        gap = best[1]["mean_val"] - second[1]["mean_val"]
        spread = max(v["std_val"] for _, v in ranked)
        print("\nval prefers %s by %+0.4f over %s" % (best[0], gap, second[0]))
        print("largest per-fold std among the configs: %.4f" % spread)
        if gap < spread:
            print("VERDICT: NOT DECIDED - the gap between configs is smaller than the spread\n"
                  "         between folds of a single config. That is the same situation as the\n"
                  "         78-image val on breast, just measured honestly. Do not promote.")
        else:
            print("VERDICT: %s wins - the gap exceeds the fold-to-fold spread, so it is not\n"
                  "         an artefact of which images landed in which fold." % best[0])
        print("pooled held-out images per config: %d"
              % max(v["pooled_val_images"] for _, v in ranked))

    out = os.path.join(MODEL_DIR, "_kfold_summary.json")
    prev = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            prev = json.load(f)
    prev.update(table)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
