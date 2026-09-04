# -*- coding: utf-8 -*-
"""
Input filter v2 — fixes the shortcut that made v1's 99.76% misleading.

THE BUG (measured by check_filter_robustness.py):
  v1 negatives = Arabic product reviews + news  -> all DECLARATIVE prose
  v1 positives = Arabic medical Q&A             -> all QUESTIONS
  So v1 could score 99.76% on its own test split by learning "is this a question?"
  rather than "is this medical?". On question-form NON-medical text it collapsed:
  rejection fell from a claimed 99.79% to a measured 33.3% (8/12 leaked through).

THE FIX — add real question-form non-medical Arabic HARD NEGATIVES:
  * hsseinmz/arcd  — Arabic Reading Comprehension Dataset (Wikipedia questions)
  * google/xquad   — xquad.ar (Wikipedia questions)
  Both are genuine Arabic questions about general knowledge, i.e. exactly the
  distribution v1 never saw. No synthetic/fabricated text is used.

We also report BOTH numbers: the in-distribution score AND the held-out hard-negative
rejection rate, so the metric can't flatter itself again.

Saves models/ar/filter.joblib + models/ar/filter_metrics.json (v1 backed up by caller).
"""
import os, json, time
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
N_NEG_SOFT = 40000     # reviews + news (declarative)
HARD_HOLDOUT = 0.25    # fraction of hard negatives held out to measure the real rate


def _text_col(df):
    cand = [c for c in df.columns if df[c].dtype == object]
    return max(cand, key=lambda c: df[c].astype(str).str.len().mean())


def load_soft_negatives():
    """v1's negatives: declarative non-medical prose."""
    frames = []
    try:
        import pyarrow.parquet as pq
        p = hf_hub_download("arbml/arabic_100k_reviews", "data/train-00000-of-00001.parquet", repo_type="dataset")
        df = pq.read_table(p).to_pandas()
        frames.append(df[_text_col(df)].astype(str))
    except Exception as e:
        print("  reviews skipped:", str(e)[:70])
    try:
        p = hf_hub_download("mksaad/Arabic_news", "Arabic_news.csv", repo_type="dataset")
        df = pd.read_csv(p)
        frames.append(df[_text_col(df)].astype(str))
    except Exception as e:
        print("  news skipped:", str(e)[:70])
    return pd.concat(frames, ignore_index=True) if frames else pd.Series(dtype=str)


def load_hard_negatives():
    """NEW: real Arabic QUESTIONS about non-medical topics (Wikipedia-based QA)."""
    import pyarrow.parquet as pq
    qs = []
    for repo, files in [("hsseinmz/arcd", ["plain_text/train-00000-of-00001.parquet",
                                           "plain_text/validation-00000-of-00001.parquet"]),
                        ("google/xquad", ["xquad.ar/validation-00000-of-00001.parquet"])]:
        for fn in files:
            try:
                p = hf_hub_download(repo, fn, repo_type="dataset")
                t = pq.read_table(p).to_pandas()
                col = "question" if "question" in t.columns else _text_col(t)
                qs.append(t[col].astype(str))
                print(f"  hard negatives: {repo}/{os.path.basename(fn)} -> {len(t)}")
            except Exception as e:
                print(f"  {repo}/{fn} skipped:", str(e)[:70])
    return pd.concat(qs, ignore_index=True) if qs else pd.Series(dtype=str)


def build_model():
    return Pipeline([
        ("feats", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=80000, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     max_features=80000, sublinear_tf=True)),
        ])),
        ("clf", SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=40, tol=1e-4,
                              class_weight="balanced", random_state=SEED)),
    ])


def main():
    t0 = time.time()
    pos = pd.read_parquet(os.path.join(DATA, "corpus.parquet"))["text"].dropna()
    pos = pos.sample(min(N_POS, len(pos)), random_state=SEED)

    soft = load_soft_negatives().map(normalize_ar)
    soft = soft[soft.str.len() >= 15].drop_duplicates()
    soft = soft.sample(min(N_NEG_SOFT, len(soft)), random_state=SEED)

    hard = load_hard_negatives().map(normalize_ar)
    hard = hard[hard.str.len() >= 15].drop_duplicates()
    print(f"[filter v2] positives={len(pos)} soft_neg={len(soft)} hard_neg={len(hard)}")
    if len(hard) < 200:
        raise SystemExit("Not enough hard negatives downloaded — aborting rather than shipping the old bug.")

    # Hold out a slice of hard negatives NEVER seen in training -> the honest number.
    hard_tr, hard_te = train_test_split(hard, test_size=HARD_HOLDOUT, random_state=SEED)

    X = pd.concat([pos, soft, pd.Series(hard_tr)], ignore_index=True).tolist()
    y = np.array([1] * len(pos) + [0] * (len(soft) + len(hard_tr)))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)

    model = build_model().fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()

    # the number that actually matters: rejection on UNSEEN question-form non-medical text
    hard_p = model.predict_proba(list(hard_te))[:, 1]
    hard_reject = float((hard_p < 0.5).mean())

    metrics = {
        "model": "arabic_input_filter_v2 (medical vs non-medical, hard-negative trained)",
        "fix": ("v1 negatives were declarative (reviews+news) while all positives were questions, so v1 "
                "partly learned 'is it a question?'. Measured non-medical rejection on question-form text "
                "was 0.333 despite a claimed 0.998. v2 adds REAL Arabic Wikipedia questions (ARCD + XQuAD-ar) "
                "as hard negatives."),
        "n_positives": int(len(pos)),
        "n_negatives_declarative": int(len(soft)),
        "n_negatives_question_form": int(len(hard_tr)),
        "threshold": 0.5,
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "medical_precision": round(float(precision_score(yte, pred)), 4),
        "medical_recall": round(float(recall_score(yte, pred)), 4),
        "nonmedical_rejection_rate": round(float(tn / max(tn + fp, 1)), 4),
        "holdout_question_negative_rejection_rate": round(hard_reject, 4),
        "n_holdout_question_negatives": int(len(hard_te)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1),
    }
    joblib.dump({"model": model, "threshold": 0.5}, os.path.join(OUT, "filter.joblib"), compress=3)
    json.dump(metrics, open(os.path.join(OUT, "filter_metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[RESULT]", json.dumps({k: metrics[k] for k in [
        "accuracy", "medical_recall", "nonmedical_rejection_rate",
        "holdout_question_negative_rejection_rate"]}, ensure_ascii=False))
    print("FILTER_V2_DONE")


if __name__ == "__main__":
    main()
