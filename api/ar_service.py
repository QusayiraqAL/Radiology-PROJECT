# -*- coding: utf-8 -*-
"""
Arabic symptom -> disease service (loads Phase 2/3/4 artifacts).

Pipeline at inference:
  1. normalize + RULE gate (Arabic? long enough? not gibberish?)     -> flag if bad
  2. LEARNED input filter (medical vs non-medical)                    -> flag if out-of-scope
  3. ROUTER  -> specialty (+ confidence). Low confidence -> abstain (top-3 specialties)
  4. PER-CATEGORY model -> fine diagnosis (top-k). No model -> specialist referral
"""
import os
import json
import numpy as np
import joblib

from ar_utils import normalize_ar, arabic_ratio
# Imported for its side effect: joblib unpickles router.joblib / cat_*.joblib by module
# reference, so TfidfUnion + SoftVoteText must be importable here or the load fails.
import ar_models  # noqa: F401

HERE = os.path.dirname(__file__)
AR = os.path.join(HERE, "models", "ar")

# thresholds (calibrated on validation; configurable)
MIN_LEN = 15
MIN_ARABIC_RATIO = 0.45
FILTER_THRESHOLD = 0.5
ROUTER_ABSTAIN = 0.30      # below this top-specialty confidence -> low-confidence abstain
DIAG_ABSTAIN = 0.35        # below this diagnosis confidence -> uncertain

_router = None
_cats = {}
_filter = None
_meta = {}
_filter_meta = {}
CANON_AR = {}
_marbert = None            # optional transformer router (GPU-trained); None -> linear router


def _try_load_marbert():
    """Load models/ar/marbert_router/ if it exists AND transformers+torch are importable.

    ar_train_transformer.py documents that the API picks this up automatically, so it has to
    actually happen here. Everything is optional: a missing folder, missing transformers, or
    a corrupt checkpoint all fall back to the linear router rather than breaking the service.
    """
    d = os.path.join(AR, "marbert_router")
    if not os.path.isdir(d):
        return None
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        meta_p = os.path.join(d, "router_meta.json")
        labels = json.load(open(meta_p, encoding="utf-8"))["labels"] if os.path.exists(meta_p) else None
        tok = AutoTokenizer.from_pretrained(d)
        mdl = AutoModelForSequenceClassification.from_pretrained(d)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(dev).eval()
        if labels is None:   # fall back to the label order baked into the config
            labels = [mdl.config.id2label[i] for i in range(mdl.config.num_labels)]
        print(f"[ar] MARBERT router loaded on {dev} ({len(labels)} specialties)")
        return {"tok": tok, "model": mdl, "labels": labels, "device": dev, "torch": torch}
    except Exception as e:
        print(f"[ar] marbert_router/ present but not usable ({type(e).__name__}: {str(e)[:70]}) "
              f"-> falling back to the linear router")
        return None


def _router_proba(text):
    """Return (classes, probabilities) from whichever router is active."""
    if _marbert is not None:
        torch = _marbert["torch"]
        enc = _marbert["tok"]([text], truncation=True, max_length=128, return_tensors="pt")
        enc = {k: v.to(_marbert["device"]) for k, v in enc.items()}
        with torch.no_grad():
            logits = _marbert["model"](**enc).logits[0]
            p = torch.softmax(logits, -1).cpu().numpy()
        return _marbert["labels"], p
    return list(_router.classes_), _router.predict_proba([text])[0]


def load():
    global _router, _cats, _filter, _meta, _filter_meta, CANON_AR, _marbert
    if not os.path.exists(os.path.join(AR, "router.joblib")):
        return False
    _router = joblib.load(os.path.join(AR, "router.joblib"))
    _marbert = _try_load_marbert()
    _meta = json.load(open(os.path.join(AR, "ar_metrics.json"), encoding="utf-8"))
    CANON_AR = _meta.get("categories", {})
    for cat in _meta.get("categories_with_fine_model", []):
        p = os.path.join(AR, f"cat_{cat}.joblib")
        if os.path.exists(p):
            _cats[cat] = joblib.load(p)
    fp = os.path.join(AR, "filter.joblib")
    if os.path.exists(fp):
        _filter = joblib.load(fp)
    fm = os.path.join(AR, "filter_metrics.json")
    if os.path.exists(fm):
        _filter_meta = json.load(open(fm, encoding="utf-8"))
    return True


def available():
    return _router is not None


def _flag(reason_ar, kind):
    return {"type": "text_ar", "ok": False, "flag": kind, "message_ar": reason_ar}


def predict(raw_text: str):
    text = normalize_ar(raw_text or "")
    # --- 1) rule gate ---
    if arabic_ratio(raw_text) < MIN_ARABIC_RATIO:
        return _flag("الرجاء كتابة وصف الأعراض باللغة العربية.", "not_arabic")
    if len(text) < MIN_LEN:
        return _flag("النص قصير جداً — الرجاء وصف الأعراض بتفصيل أكبر.", "too_short")
    toks = text.split()
    if len(set(toks)) < 3:
        return _flag("النص غير واضح أو متكرر — صف الأعراض بجُمَل مفهومة.", "gibberish")

    # --- 2) learned input filter (medical vs non-medical) ---
    filter_p = None
    if _filter is not None:
        filter_p = float(_filter["model"].predict_proba([text])[0][1])
        if filter_p < _filter.get("threshold", FILTER_THRESHOLD):
            r = _flag("لا يبدو هذا النص وصفاً لأعراض طبية — الرجاء إدخال شكوى صحية واضحة.", "not_medical")
            r["medical_score"] = round(filter_p, 3)
            return r

    # --- 3) specialty router (MARBERT if trained, else the linear model) ---
    classes, proba = _router_proba(text)
    order = np.argsort(proba)[::-1]
    top_cat = classes[order[0]]
    top_conf = float(proba[order[0]])
    top3 = [{"category": classes[i], "category_ar": CANON_AR.get(classes[i], classes[i]),
             "confidence": round(float(proba[i]) * 100, 1)} for i in order[:3]]

    low_conf = top_conf < ROUTER_ABSTAIN

    # --- 4) per-category fine diagnosis ---
    diagnosis = None
    cat_model = _cats.get(top_cat)
    cat_acc = _meta.get("per_category", {}).get(top_cat, {}).get("accuracy")
    if cat_model is not None:
        dp = cat_model.predict_proba([text])[0]
        dcls = list(cat_model.classes_)
        dorder = np.argsort(dp)[::-1]
        diagnosis = {
            "top": dcls[dorder[0]],
            "confidence": round(float(dp[dorder[0]]) * 100, 1),
            "uncertain": bool(dp[dorder[0]] < DIAG_ABSTAIN),
            "candidates": [{"name_ar": dcls[i], "probability": round(float(dp[i]) * 100, 1)}
                           for i in dorder[:4]],
            "model_accuracy": cat_acc,
        }

    return {
        "type": "text_ar", "ok": True,
        "flag": "low_confidence" if low_conf else None,
        "medical_score": round(filter_p, 3) if filter_p is not None else None,
        "specialty": top_cat,
        "specialty_ar": CANON_AR.get(top_cat, top_cat),
        "specialty_confidence": round(top_conf * 100, 1),
        "top3_specialties": top3,
        "diagnosis": diagnosis,
        "has_fine_model": cat_model is not None,
        "message_ar": ("الثقة منخفضة — قد تحتاج الحالة لمزيد من التفاصيل أو مراجعة طبيب مختص."
                       if low_conf else None),
    }


def router_backend():
    """Which router is actually serving: the GPU-trained transformer or the linear model."""
    return "marbert" if _marbert is not None else "linear"


def meta():
    backend = router_backend()
    return {
        "id": "symptoms_ar",
        "title_ar": "التنبؤ بالأمراض من الأعراض (نص عربي)",
        "title_en": "Arabic symptom-text disease prediction",
        "modality": "clinical text (Arabic)",
        "kind": "text_ar",
        "available": available(),
        "router_backend": backend,
        "source": (f"Router(20 specialties, {backend}) + per-category models + input filter "
                   "— trained on >=200k real Arabic entries"),
        "metrics": {**_meta, "filter": _filter_meta} if _meta else None,
    }
