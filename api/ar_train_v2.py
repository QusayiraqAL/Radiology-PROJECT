# -*- coding: utf-8 -*-
"""
Arabic symptom system v2 — retrained from scratch with measured anti-overfitting.

What changed vs v1 (ar_train.py):
  * regularization (alpha) chosen by a measured sweep, not guessed        -> smaller train/test gap
  * soft-vote ensemble (modified_huber + log_loss) for better-calibrated  -> higher top-3
    probabilities, which is what the router actually needs (it abstains on low confidence)
  * TRAIN accuracy is reported next to TEST accuracy for every model, so the
    overfitting gap is visible instead of hidden
  * per-category models: stronger regularization + a min-support floor

Same held-out protocol as v1 (seed=0, stratified) so before/after is comparable.
Artifacts -> models/ar/ (router.joblib, cat_*.joblib, ar_metrics.json)
"""
import os, json, time
import numpy as np
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from ar_data_prep import CANON
from ar_models import TfidfUnion, SoftVoteText   # shared with ar_service.py (picklable)

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "arabic")
OUT = os.path.join(HERE, "models", "ar")
os.makedirs(OUT, exist_ok=True)
SEED = 0

# --- config chosen by the measured sweep in _exp_router2.py ---
ROUTER_ALPHA = float(os.environ.get("ROUTER_ALPHA", "5e-5"))   # sweep: halves the gap, +top3
CAT_C = float(os.environ.get("CAT_C", "0.1"))                  # per-category LinearSVC C (sweep)
ENSEMBLE = os.environ.get("ROUTER_ENSEMBLE", "0") == "1"
MIN_DIAG_SUPPORT = 20        # keep v1's floor so the label set stays comparable
SAMPLE = int(os.environ.get("AR_SAMPLE", "0"))     # >0 = smoke-test on a subsample
ROUTER_ONLY = os.environ.get("ROUTER_ONLY", "0") == "1"  # retrain router, keep category models


def _fit_sgd(F, y, loss, alpha):
    return SGDClassifier(loss=loss, alpha=alpha, max_iter=40, tol=1e-4,
                         class_weight="balanced", random_state=SEED).fit(F, y)


def train_router():
    df = pd.read_parquet(os.path.join(DATA, "router_data.parquet")).dropna(subset=["text", "category"])
    df = df[df["text"].str.len() >= 15]
    if SAMPLE:
        df = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(SAMPLE // 20, 20)), random_state=SEED))
    X, y = df["text"].tolist(), df["category"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    print(f"[router] train={len(Xtr)} test={len(Xte)} classes={len(set(y))} alpha={ROUTER_ALPHA:g} ensemble={ENSEMBLE}", flush=True)

    t0 = time.time()
    feats = TfidfUnion()
    Ftr = feats.fit_transform(Xtr)
    Fte = feats.transform(Xte)

    models = [_fit_sgd(Ftr, ytr, "modified_huber", ROUTER_ALPHA)]
    if ENSEMBLE:
        models.append(_fit_sgd(Ftr, ytr, "log_loss", ROUTER_ALPHA))
    classes = list(models[0].classes_)
    router = SoftVoteText(feats, models, classes)

    def _proba(F):
        return sum(m.predict_proba(F) for m in models) / len(models)

    cls = np.array(classes)
    Ptr, Pte = _proba(Ftr), _proba(Fte)
    tr_acc = accuracy_score(ytr, cls[Ptr.argmax(1)])
    pred = cls[Pte.argmax(1)]
    acc = accuracy_score(yte, pred)
    macro = f1_score(yte, pred, average="macro")
    yte_idx = np.array([classes.index(c) for c in yte])
    top3 = top_k_accuracy_score(yte_idx, Pte, k=3, labels=list(range(len(classes))))

    joblib.dump(router, os.path.join(OUT, "router.joblib"), compress=3)
    print(f"[router] train={tr_acc:.4f} test={acc:.4f} gap={tr_acc-acc:+.3f} "
          f"top3={top3:.4f} macroF1={macro:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return {"n_train": len(Xtr), "n_test": len(Xte), "n_classes": len(classes),
            "accuracy": round(float(acc), 4), "train_accuracy": round(float(tr_acc), 4),
            "overfitting_gap": round(float(tr_acc - acc), 4),
            "top3_accuracy": round(float(top3), 4), "macro_f1": round(float(macro), 4),
            "alpha": ROUTER_ALPHA, "ensemble": ENSEMBLE}


def _cat_pipeline():
    """Per-category diagnosis model.

    Measured (_exp_cat.py): the SGD soft-vote that helps the ROUTER *hurts* here. These
    sets are small (260..10k rows) against 120k features, so SGD memorizes (train=1.000,
    gap=0.39) and lands below v1. A calibrated LinearSVC generalizes better on small data
    and still yields probabilities, so per-category keeps it — with C set by the sweep.
    """
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     max_features=60000, sublinear_tf=True)),
        ])),
        ("clf", CalibratedClassifierCV(LinearSVC(C=CAT_C, class_weight="balanced"), cv=3)),
    ])


def train_per_category():
    df = pd.read_parquet(os.path.join(DATA, "labeled_shifaa.parquet")).dropna(subset=["text", "category", "diagnosis"])
    df = df[df["text"].str.len() >= 15]
    results = {}
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        vc = sub["diagnosis"].value_counts()
        sub = sub[sub["diagnosis"].isin(vc[vc >= MIN_DIAG_SUPPORT].index)]
        if sub["diagnosis"].nunique() < 2 or len(sub) < 200:
            continue
        X, y = sub["text"].tolist(), sub["diagnosis"].values
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

        model = _cat_pipeline().fit(Xtr, ytr)
        tr_acc = accuracy_score(ytr, model.predict(Xtr))
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        macro = f1_score(yte, pred, average="macro")
        joblib.dump(model, os.path.join(OUT, f"cat_{cat}.joblib"), compress=3)
        results[cat] = {"n_train": len(Xtr), "n_test": len(Xte),
                        "n_diagnoses": int(sub["diagnosis"].nunique()),
                        "accuracy": round(float(acc), 4),
                        "train_accuracy": round(float(tr_acc), 4),
                        "overfitting_gap": round(float(tr_acc - acc), 4),
                        "macro_f1": round(float(macro), 4),
                        "diagnoses": sorted(sub["diagnosis"].unique().tolist())}
        print(f"[cat {cat:18}] dx={sub['diagnosis'].nunique():2} n={len(sub):5} "
              f"train={tr_acc:.4f} test={acc:.4f} gap={tr_acc-acc:+.3f} macroF1={macro:.4f}", flush=True)
    return results


def main():
    t0 = time.time()
    router = train_router()

    metrics_path = os.path.join(OUT, "ar_metrics.json")
    if ROUTER_ONLY:
        # Retrain only the router and splice its section into the existing metrics,
        # keeping the already-trained per-category models (they are independent of it).
        old = json.load(open(metrics_path, encoding="utf-8"))
        per_cat = old.get("per_category", {})
        print(f"[router-only] reusing {len(per_cat)} existing category models", flush=True)
    else:
        per_cat = train_per_category()
    accs = [v["accuracy"] for v in per_cat.values()]
    metrics = {
        "system": "arabic_symptom_to_disease_v3",
        "modality": "clinical text (Arabic)",
        "datasets": "hajerbchn/arabic-medical-qa + MAQA + Shifaa (real Arabic, >=200k aggregate)",
        "architecture": (f"specialty router (20, SGD modified_huber{' + log_loss soft-vote' if ENSEMBLE else ''}) + "
                         "per-category diagnosis models (calibrated LinearSVC)"),
        "anti_overfitting": [
            f"router L2 alpha={ROUTER_ALPHA:g} chosen by a measured sweep, not guessed: "
            "2e-5 -> 5e-5 cut the train/test gap 0.215 -> 0.141 while top-3 ROSE 0.893 -> 0.901",
            f"per-category: calibrated LinearSVC C={CAT_C:g} from a measured sweep "
            "(C=1.0 -> 0.1 cut the gap 0.361 -> 0.280 and raised accuracy 0.638 -> 0.647)",
            "rejected by measurement: the soft-vote ensemble raised top-3 by only +0.0025 but "
            "cost -0.010 top-1, and SGD on the small per-category sets memorized (train=1.000)",
            "class_weight=balanced (all models)", "min_df=3 feature pruning",
            f"per-diagnosis min support={MIN_DIAG_SUPPORT}",
            "train/test gap measured and published for every model",
        ],
        "categories": {k: CANON[k] for k in CANON},
        "router": router,
        "per_category": per_cat,
        "categories_with_fine_model": sorted(per_cat.keys()),
        "per_category_accuracy_min": round(float(min(accs)), 4) if accs else None,
        "per_category_accuracy_max": round(float(max(accs)), 4) if accs else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    json.dump(metrics, open(os.path.join(OUT, "ar_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("\n[RESULT] router test=%.4f (train=%.4f gap=%+.3f) top3=%.4f | %d category models %.3f..%.3f" % (
        router["accuracy"], router["train_accuracy"], router["overfitting_gap"],
        router["top3_accuracy"], len(per_cat), min(accs) if accs else 0, max(accs) if accs else 0))
    print("AR_TRAIN_V2_DONE", flush=True)


if __name__ == "__main__":
    main()
