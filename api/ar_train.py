# -*- coding: utf-8 -*-
"""
Phase 2 — Train the category-organized Arabic system on the real >=200k corpus.

  * ROUTER            : symptom text -> 1 of 20 medical specialties   (trained on ~196k)
  * PER-CATEGORY MODEL: symptom text -> fine diagnosis within a specialty (Shifaa labels)

Held-out test splits give the measured accuracy. Models + metrics saved to models/ar/.
"""
import os
import json
import time
import numpy as np
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from ar_data_prep import CANON

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "arabic")
OUT = os.path.join(HERE, "models", "ar")
os.makedirs(OUT, exist_ok=True)
SEED = 0


def features(word_max=120000, char_max=120000):
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=word_max, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=char_max, sublinear_tf=True)),
    ])


def big_model():
    # fast + probabilistic for the large router
    return Pipeline([("feats", features()),
                     ("clf", SGDClassifier(loss="modified_huber", alpha=2e-5, max_iter=40,
                                           tol=1e-4, class_weight="balanced", random_state=SEED))])


def small_model():
    # calibrated linear SVM for the smaller per-category models
    return Pipeline([("feats", features(60000, 60000)),
                     ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3))])


def train_router():
    df = pd.read_parquet(os.path.join(DATA, "router_data.parquet")).dropna(subset=["text", "category"])
    df = df[df["text"].str.len() >= 15]
    X, y = df["text"].tolist(), df["category"].tolist()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)
    print(f"[router] train={len(Xtr)} test={len(Xte)} classes={len(set(y))}")
    t0 = time.time()
    model = big_model().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    classes = list(model.classes_)
    pred = [classes[i] for i in proba.argmax(1)]
    yte_idx = [classes.index(c) for c in yte]
    acc = accuracy_score(yte, pred)
    top3 = top_k_accuracy_score(yte_idx, proba, k=3, labels=list(range(len(classes))))
    macro = f1_score(yte, pred, average="macro")
    joblib.dump(model, os.path.join(OUT, "router.joblib"))
    print(f"[router] acc={acc:.4f} top3={top3:.4f} macroF1={macro:.4f} ({time.time()-t0:.0f}s)")
    return {"n_train": len(Xtr), "n_test": len(Xte), "n_classes": len(classes),
            "accuracy": round(float(acc), 4), "top3_accuracy": round(float(top3), 4),
            "macro_f1": round(float(macro), 4)}


def train_per_category():
    df = pd.read_parquet(os.path.join(DATA, "labeled_shifaa.parquet")).dropna(subset=["text", "category", "diagnosis"])
    df = df[df["text"].str.len() >= 15]
    results = {}
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        # drop ultra-rare diagnoses (can't be learned/measured reliably)
        vc = sub["diagnosis"].value_counts()
        keep = vc[vc >= 20].index
        sub = sub[sub["diagnosis"].isin(keep)]
        if sub["diagnosis"].nunique() < 2 or len(sub) < 200:
            continue
        X, y = sub["text"].tolist(), sub["diagnosis"].tolist()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
        model = small_model().fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        macro = f1_score(yte, pred, average="macro")
        joblib.dump(model, os.path.join(OUT, f"cat_{cat}.joblib"))
        results[cat] = {"n_train": len(Xtr), "n_test": len(Xte),
                        "n_diagnoses": int(sub["diagnosis"].nunique()),
                        "accuracy": round(float(acc), 4), "macro_f1": round(float(macro), 4),
                        "diagnoses": sorted(sub["diagnosis"].unique().tolist())}
        print(f"[cat {cat:18}] dx={sub['diagnosis'].nunique():2} n={len(sub):5} acc={acc:.4f} macroF1={macro:.4f}")
    return results


def main():
    t0 = time.time()
    router = train_router()
    per_cat = train_per_category()
    accs = [v["accuracy"] for v in per_cat.values()]
    metrics = {
        "system": "arabic_symptom_to_disease_v2",
        "modality": "clinical text (Arabic)",
        "datasets": "hajerbchn/arabic-medical-qa + MAQA + Shifaa (real Arabic, >=200k aggregate)",
        "architecture": "specialty router (20) + per-category diagnosis models (Shifaa fine labels)",
        "categories": {k: CANON[k] for k in CANON},
        "router": router,
        "per_category": per_cat,
        "categories_with_fine_model": sorted(per_cat.keys()),
        "per_category_accuracy_min": round(float(min(accs)), 4) if accs else None,
        "per_category_accuracy_max": round(float(max(accs)), 4) if accs else None,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    json.dump(metrics, open(os.path.join(OUT, "ar_metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n[RESULT] router acc=%.4f top3=%.4f | per-category %d models acc %.3f..%.3f" % (
        router["accuracy"], router["top3_accuracy"], len(per_cat),
        min(accs) if accs else 0, max(accs) if accs else 0))
    print("AR_TRAIN_DONE")


if __name__ == "__main__":
    main()
