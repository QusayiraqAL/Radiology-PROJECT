# -*- coding: utf-8 -*-
"""Aggregate every model's measured metrics into one validation_report.json + printed summary."""
import os
import json
import time

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "models")

# each entry: list of candidate files, best/newest FIRST (v2 preferred, v1 fallback)
FILES = {
    "chest": ["chest_metrics.json"],
    "pneumonia": ["pneumonia_v2_metrics.json", "pneumonia_metrics.json"],
    "brain": ["brain_v2_metrics.json", "brain_metrics.json"],
    "symptoms_ar": [os.path.join("ar", "ar_metrics.json")],
    "input_filter_ar": [os.path.join("ar", "filter_metrics.json")],
}


def _gap(m):
    """Generalization gap = train acc - test acc. Reported so overfitting is visible."""
    src = m.get("router", m)
    if src.get("train_accuracy") is None or src.get("overfitting_gap") is None:
        return ""
    return f"  [train={src['train_accuracy']} gap={src['overfitting_gap']:+.3f}]"


def main():
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}
    print("=" * 66)
    print("  VALIDATION SUMMARY — measured on held-out test sets")
    print("=" * 66)
    for mid, cands in FILES.items():
        p = next((os.path.join(MODEL_DIR, c) for c in cands
                  if os.path.exists(os.path.join(MODEL_DIR, c))), None)
        if p is None:
            print(f"  [{mid:9}] NOT TRAINED YET ({cands[0]})")
            continue
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        m["_metrics_file"] = os.path.basename(p)
        report["models"][mid] = m
        if mid == "chest":
            print(f"  [chest    ] mean ROC-AUC = {m.get('mean_auc')}  "
                  f"(n_test={m.get('n_test')}, {len(m.get('per_label_auc', {}))} labels)")
        elif mid == "pneumonia":
            print(f"  [pneumonia] acc={m.get('test_accuracy')}  AUC={m.get('test_auc')}  "
                  f"sens={m.get('test_sensitivity_recall')}  spec={m.get('test_specificity')}  "
                  f"(n_test={m.get('n_test')}){_gap(m)}")
        elif mid == "brain":
            print(f"  [brain    ] acc={m.get('test_accuracy')}  macroF1={m.get('test_macro_f1')}  "
                  f"macroAUC={m.get('test_auc_macro_ovr')}  (n_test={m.get('n_test')}){_gap(m)}")
            if m.get("split_note"):
                print(f"              split: {m.get('split', '')[:88]}")
        elif mid == "symptoms_ar":
            r = m.get("router", {})
            print(f"  [symptoms_ar] router_acc={r.get('accuracy')}  top3={r.get('top3_accuracy')}  "
                  f"per-cat={m.get('per_category_accuracy_min')}..{m.get('per_category_accuracy_max')}  "
                  f"(router n_test={r.get('n_test')}, {len(m.get('per_category', {}))} category models)"
                  f"{_gap(m)}")
        elif mid == "input_filter_ar":
            # v2 splits negatives into declarative + question-form; v1 had a single n_negatives
            n_neg = m.get("n_negatives")
            if n_neg is None:
                n_neg = (m.get("n_negatives_declarative", 0) or 0) + (m.get("n_negatives_question_form", 0) or 0)
            print(f"  [input_filter_ar] acc={m.get('accuracy')}  reject_nonmedical={m.get('nonmedical_rejection_rate')}  "
                  f"(n={m.get('n_positives')}+{n_neg})")
            hard = m.get("holdout_question_negative_rejection_rate")
            if hard is not None:
                # the number that matters: rejection on question-form text it never trained on
                print(f"                    reject_unseen_question_negatives={hard}  "
                      f"(n={m.get('n_holdout_question_negatives')})  <- the honest one")
    out = os.path.join(MODEL_DIR, "validation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("=" * 66)
    print("  Written:", out)


if __name__ == "__main__":
    main()
