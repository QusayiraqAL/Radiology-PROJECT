# -*- coding: utf-8 -*-
"""
Phase 3 — Input-filtering / quality gate for the Arabic text system.

Goal: flag "unsatisfactory" inputs instead of forcing a diagnosis.
  (a) Rule checks (done in the API at serve time): must be Arabic, long enough, not gibberish.
  (b) Learned gate (here): a medical-symptom vs non-medical classifier.
        positives = real Arabic medical corpus (>=200k pool)
        negatives = general Arabic text (reviews + news) — clearly NOT medical symptoms
      If P(medical) < threshold -> input is flagged as out-of-scope.

Saves models/ar/filter.joblib + models/ar/filter_metrics.json.
"""
import os
import json
import time
import numpy as np
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from ar_utils import normalize_ar

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "arabic")
OUT = os.path.join(HERE, "models", "ar")
os.makedirs(OUT, exist_ok=True)
SEED = 0
N_POS = 50000
N_NEG = 50000


def _text_col(df):
    cand = [c for c in df.columns if df[c].dtype == object]
    return max(cand, key=lambda c: df[c].astype(str).str.len().mean())


def load_negatives():
    frames = []
    # Arabic product/app reviews (non-medical everyday language)
    try:
        import pyarrow.parquet as pq
        p = hf_hub_download("arbml/arabic_100k_reviews", "data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pq.read_table(p).to_pandas()
        col = _text_col(df)
        frames.append(df[col].astype(str))
    except Exception as e:
        print("  reviews skipped:", str(e)[:70])
    # Arabic news
    try:
        p = hf_hub_download("mksaad/Arabic_news", "Arabic_news.csv", repo_type="dataset")
        df = pd.read_csv(p)
        col = _text_col(df)
        frames.append(df[col].astype(str))
    except Exception as e:
        print("  news skipped:", str(e)[:70])
    neg = pd.concat(frames, ignore_index=True)
    return neg


def main():
    t0 = time.time()
    pos = pd.read_parquet(os.path.join(DATA, "corpus.parquet"))["text"].dropna()
    pos = pos.sample(min(N_POS, len(pos)), random_state=SEED)

    neg_raw = load_negatives().map(normalize_ar)
    neg_raw = neg_raw[neg_raw.str.len() >= 15].drop_duplicates()
    neg = neg_raw.sample(min(N_NEG, len(neg_raw)), random_state=SEED)
    print(f"[filter] positives={len(pos)} negatives={len(neg)}")

    X = pd.concat([pos, neg], ignore_index=True).tolist()
    y = np.array([1] * len(pos) + [0] * len(neg))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)

    model = Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=80000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=80000, sublinear_tf=True)),
        ])),
        ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=40, tol=1e-4, random_state=SEED)),
    ]).fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    metrics = {
        "model": "arabic_input_filter (medical vs non-medical)",
        "n_positives": int(len(pos)), "n_negatives": int(len(neg)),
        "threshold": 0.5,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "medical_precision": round(float(precision_score(yte, pred)), 4),
        "medical_recall": round(float(recall_score(yte, pred)), 4),
        "nonmedical_rejection_rate": round(float(tn / max(tn + fp, 1)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    joblib.dump({"model": model, "threshold": 0.5}, os.path.join(OUT, "filter.joblib"))
    json.dump(metrics, open(os.path.join(OUT, "filter_metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[RESULT]", json.dumps(metrics, ensure_ascii=False))
    print("FILTER_DONE")


if __name__ == "__main__":
    main()
