#!/usr/bin/env python
"""Ordinal decision rule for a graded task: cut an expected-grade score with tuned thresholds.

Usage:  python ordinal_rounding.py retina

Taken from the Diabetic Retinopathy notebooks in `code benifit/` (all three use it; the
OptimizedRounder there credits kaggle.com/abhishek/optimizer-for-quadratic-weighted-kappa).
Those notebooks train a REGRESSION head and cut its scalar output at four learned thresholds.
We already serve a 5-way softmax, so the scalar comes from it instead:

    score = sum(k * p_k)        the expected grade under the model's own distribution

That keeps the whole thing training-free - it is a decision rule over probabilities this
project already produces, not a new model.

Why this is not the CORAL attempt of session 3 step 5 (0.6075 -> 0.5000): CORAL replaces the
head with K-1 tied binary classifiers and retrains. This changes nothing about the network. If
it helps, it helps for free; if it does not, it cost one CPU pass.

Argmax throws away the ordering: predicting grade 4 for a grade-0 eye and predicting grade 1
for it are the same single error to a softmax. The expected grade keeps that information, and
the thresholds decide where the cuts go.

DISCIPLINE: thresholds are fitted on VAL only. Test is scored once, with the val-fitted cuts.
Fitting them on test would report max-over-thresholds of a test number, which is the same
test-set-selection bug the trainer's TTA decision had (step 21).
"""
import json, os, sys
import numpy as np
from functools import partial
from scipy import optimize
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ensemble_archs import member_probs
from check_serving_path import served_checkpoints   # manifest, then _v2, then base
from verify_retrain_gains import MODEL_DIR


def cut(score, coef):
    """Bucket a scalar score into 0..len(coef) using ascending cut points."""
    return np.searchsorted(np.sort(coef), score)


def grid_cuts(score, y, n_cls, objective, step=0.1):
    """Exhaustive ascending-cut search. Slow and dumb, and it cannot stall on a plateau.

    Nelder-Mead on a piecewise-constant objective can return its starting point and call it a
    minimum - it did, on the accuracy objective. This is here to tell an unbeatable rule apart
    from an optimiser that never moved (TRAINING_LOG step 101).
    """
    import itertools
    fns = {"acc": lambda yt, yp: accuracy_score(yt, yp),
           "qwk": lambda yt, yp: cohen_kappa_score(yt, yp, weights="quadratic")}
    f = fns[objective]
    lo, hi = float(score.min()), float(score.max())
    grid = np.arange(lo, hi + step, step)
    best, best_c = -1.0, None
    for combo in itertools.combinations(grid, n_cls - 1):
        v = f(y, cut(score, np.asarray(combo)))
        if v > best:
            best, best_c = v, np.asarray(combo)
    return np.sort(best_c), best


def fit_cuts(score, y, n_cls, objective):
    """Nelder-Mead on the cut points, exactly as the notebooks do, but with a chosen objective.

    The notebooks optimise quadratic-weighted kappa because that is the DR competition metric.
    This project's headline is accuracy, so both are fitted and both are reported - optimising
    one and quoting the other is how a technique gets credited for a gain it did not produce.
    """
    fns = {"acc": lambda yt, yp: accuracy_score(yt, yp),
           "qwk": lambda yt, yp: cohen_kappa_score(yt, yp, weights="quadratic")}
    f = fns[objective]
    loss = lambda c: -f(y, cut(score, c))
    init = np.arange(n_cls - 1) + 0.5              # 0.5, 1.5, 2.5, 3.5 for 5 grades
    res = optimize.minimize(loss, init, method="nelder-mead")
    return np.sort(res["x"])


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else "retina"
    # Resolve what main.py would load, not models/<key>.pt. For retina those are different
    # models two sessions apart - see the note in member_probs.
    paths, is_ens = served_checkpoints(key)
    if is_ens or len(paths) != 1:
        raise SystemExit("%s is served as an ensemble of %d; this rule needs one probability "
                         "matrix, so average the members first" % (key, len(paths)))
    print("served checkpoint: %s" % os.path.basename(paths[0]))
    pv, yv, meta = member_probs(paths[0], "val")
    pt, yt, _ = member_probs(paths[0], "test")
    n_cls = pv.shape[1]
    grades = np.arange(n_cls)
    sv, st = pv @ grades, pt @ grades              # expected grade

    base_v, base_t = pv.argmax(1), pt.argmax(1)
    print("%s  arch=%s  n_val=%d  n_test=%d  classes=%d" % (key, meta["arch"], len(yv), len(yt), n_cls))
    print("\n%-26s %-9s %-9s %-9s %-9s" % ("rule", "val_acc", "val_qwk", "test_acc", "test_qwk"))

    def row(label, pv_, pt_):
        print("%-26s %-9.4f %-9.4f %-9.4f %-9.4f"
              % (label, accuracy_score(yv, pv_), cohen_kappa_score(yv, pv_, weights="quadratic"),
                 accuracy_score(yt, pt_), cohen_kappa_score(yt, pt_, weights="quadratic")))
        return {"val_accuracy": round(float(accuracy_score(yv, pv_)), 4),
                "val_qwk": round(float(cohen_kappa_score(yv, pv_, weights="quadratic")), 4),
                "test_accuracy": round(float(accuracy_score(yt, pt_)), 4),
                "test_qwk": round(float(cohen_kappa_score(yt, pt_, weights="quadratic")), 4)}

    out = {"key": key, "arch": meta["arch"], "n_val": int(len(yv)), "n_test": int(len(yt)),
           "rules": {}}
    out["rules"]["argmax"] = row("argmax (served)", base_v, base_t)
    out["rules"]["expected_grade_default"] = row("expected grade, cuts .5", cut(sv, np.arange(n_cls - 1) + .5),
                                                                            cut(st, np.arange(n_cls - 1) + .5))
    for obj in ("acc", "qwk"):
        c = fit_cuts(sv, yv, n_cls, obj)
        out["rules"]["expected_grade_fit_" + obj] = row("expected grade, fit %s" % obj, cut(sv, c), cut(st, c))
        out["rules"]["expected_grade_fit_" + obj]["cuts"] = [round(float(x), 4) for x in c]
        print("%-26s cuts = %s" % ("", np.round(c, 3).tolist()))

    print("\nconfusion, argmax (test):")
    for r in confusion_matrix(yt, base_t).tolist():
        print("   ", r)
    for obj in ("acc", "qwk"):
        c, v = grid_cuts(sv, yv, n_cls, obj)
        out["rules"]["expected_grade_grid_" + obj] = row("expected grade, GRID %s" % obj,
                                                         cut(sv, c), cut(st, c))
        out["rules"]["expected_grade_grid_" + obj]["cuts"] = [round(float(x), 3) for x in c]
        print("%-26s cuts = %s   best val %s = %.4f" % ("", np.round(c, 3).tolist(), obj, v))

    best = max([k for k in out["rules"] if k != "argmax"],
               key=lambda k: out["rules"][k]["val_accuracy"])
    print("")
    print("val picks %s (val_acc %.4f) vs argmax %.4f"
          % (best, out["rules"][best]["val_accuracy"], out["rules"]["argmax"]["val_accuracy"]))
    p = os.path.join(MODEL_DIR, "_ordinal_rounding.json")
    prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    prev[key] = out
    json.dump(prev, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("wrote models/_ordinal_rounding.json")


if __name__ == "__main__":
    main()
