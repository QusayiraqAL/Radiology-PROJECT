# -*- coding: utf-8 -*-
"""
Train a REAL text-based disease-prediction system, organized by category.

Dataset: gretelai/symptom_to_diagnosis (real natural-language symptom descriptions
-> 22 diagnoses, official train/test split).

Architecture (hierarchical, "dedicated pre-trained model per category"):
  * ROUTER model:  symptom text -> disease category
  * one PER-CATEGORY model:  symptom text -> specific disease within that category

Everything is trained + validated on the held-out TEST split. Only measured
accuracy is reported.
"""
import os
import json
import time
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score

from text_meta import CATEGORY_MAP

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "symptom_text")
OUT = os.path.join(HERE, "models", "text")
os.makedirs(OUT, exist_ok=True)
SEED = 0


def load(split):
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"{split}.jsonl"), encoding="utf-8") if l.strip()]
    X = [r["input_text"] for r in rows]
    y = [r["output_text"].strip().lower() for r in rows]
    return X, y


def make_model():
    """TF-IDF (word 1-2gram + char 2-5gram) + calibrated LinearSVC (accuracy + probabilities)."""
    from sklearn.pipeline import FeatureUnion
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
    ])
    clf = CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)
    return Pipeline([("feats", feats), ("clf", clf)])


def main():
    t0 = time.time()
    Xtr, ytr_dis = load("train")
    Xte, yte_dis = load("test")
    ytr_cat = [CATEGORY_MAP[d] for d in ytr_dis]
    yte_cat = [CATEGORY_MAP[d] for d in yte_dis]
    categories = sorted(set(CATEGORY_MAP.values()))
    print(f"[data] train={len(Xtr)} test={len(Xte)} diseases={len(set(ytr_dis))} categories={len(categories)}")

    # ---- 1) ROUTER: text -> category ----
    router = make_model().fit(Xtr, ytr_cat)
    router_pred = router.predict(Xte)
    router_acc = accuracy_score(yte_cat, router_pred)
    joblib.dump(router, os.path.join(OUT, "router.joblib"))
    print(f"[router] test accuracy = {router_acc:.4f}")

    # ---- 2) PER-CATEGORY models: text -> disease within category ----
    per_category = {}
    for cat in categories:
        idx = [i for i, c in enumerate(ytr_cat) if c == cat]
        Xc = [Xtr[i] for i in idx]
        yc = [ytr_dis[i] for i in idx]
        diseases = sorted(set(yc))
        model = make_model().fit(Xc, yc)
        joblib.dump(model, os.path.join(OUT, f"cat_{cat}.joblib"))

        # evaluate on the TEST rows that truly belong to this category
        tidx = [i for i, c in enumerate(yte_cat) if c == cat]
        Xt = [Xte[i] for i in tidx]
        yt = [yte_dis[i] for i in tidx]
        pred = model.predict(Xt)
        acc = accuracy_score(yt, pred)
        per_category[cat] = {
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
            "n_train": len(Xc), "n_test": len(Xt),
            "diseases": diseases,
        }
        print(f"[cat {cat:22}] diseases={len(diseases)} test_acc={acc:.4f} (n_test={len(Xt)})")

    # ---- 3) END-TO-END: route then predict disease ----
    e2e_correct = 0
    for i, x in enumerate(Xte):
        cat = router.predict([x])[0]
        model = joblib.load(os.path.join(OUT, f"cat_{cat}.joblib"))
        dis = model.predict([x])[0]
        e2e_correct += int(dis == yte_dis[i])
    e2e_acc = e2e_correct / len(Xte)

    # a flat 22-class reference model for context
    flat = make_model().fit(Xtr, ytr_dis)
    flat_acc = accuracy_score(yte_dis, flat.predict(Xte))
    joblib.dump(flat, os.path.join(OUT, "flat_reference.joblib"))

    accs = [v["accuracy"] for v in per_category.values()]
    metrics = {
        "system": "text_symptom_to_disease",
        "modality": "clinical text (symptom description)",
        "dataset": "gretelai/symptom_to_diagnosis (real NL symptom text, 22 diagnoses)",
        "architecture": "hierarchical: category router + one dedicated model per category",
        "n_train": len(Xtr), "n_test": len(Xte),
        "n_diseases": len(set(ytr_dis)), "n_categories": len(categories),
        "router_accuracy": round(float(router_acc), 4),
        "end_to_end_accuracy": round(float(e2e_acc), 4),
        "flat_reference_accuracy": round(float(flat_acc), 4),
        "per_category_accuracy_min": round(float(min(accs)), 4),
        "per_category_accuracy_max": round(float(max(accs)), 4),
        "per_category": per_category,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "text_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n[RESULT]")
    print(f"  router acc         = {router_acc:.4f}")
    print(f"  end-to-end acc     = {e2e_acc:.4f}")
    print(f"  flat 22-class acc  = {flat_acc:.4f}")
    print(f"  per-category range = {min(accs):.4f} .. {max(accs):.4f}")
    print("TEXT_TRAIN_DONE")


if __name__ == "__main__":
    main()
