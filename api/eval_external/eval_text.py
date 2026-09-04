# -*- coding: utf-8 -*-
"""EXTERNAL-ish validation for the Arabic router.

Public Arabic medical-QA sets with a mappable 20-specialty taxonomy were all gated (HTTP 401)
when probed, so a brand-new external cohort wasn't obtainable. Instead we use the HELD-OUT
test split (seed=0, 15%) that the router never trained on, and sample >=2500 stratified.
These rows were excluded from fitting, so they are genuinely unseen by the model — the caveat
is only that they share the training distribution (not a different source).
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import save_results, API
import ar_service
from ar_utils import normalize_ar

SEED, N = 0, 2500
DATA = os.path.join(API, "data", "arabic", "router_data.parquet")


def main():
    assert ar_service.load(), "Arabic system not loaded"
    df = pd.read_parquet(DATA).dropna(subset=["text", "category"])
    df = df[df["text"].str.len() >= 15]
    X, y = df["text"].tolist(), df["category"].values
    # SAME split as training -> take the held-out TEST portion only
    _, Xte, _, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=SEED)

    te = pd.DataFrame({"text": Xte, "cat": yte})
    n_per = max(N // te["cat"].nunique(), 20)
    samp = te.groupby("cat", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n_per), random_state=SEED))
    print(f"[text] held-out pool={len(te)} -> sampled {len(samp)} across {samp['cat'].nunique()} specialties", flush=True)

    backend = ar_service.router_backend()
    texts = [normalize_ar(t) for t in samp["text"].tolist()]
    t0 = time.time()
    if backend == "marbert":
        mb = ar_service._marbert
        classes = list(mb["labels"])
        torch = mb["torch"]
        proba = np.zeros((len(texts), len(classes)), dtype=np.float32)
        for i in range(0, len(texts), 64):
            enc = mb["tok"](texts[i:i+64], truncation=True, max_length=128,
                            padding=True, return_tensors="pt")
            enc = {k: v.to(mb["device"]) for k, v in enc.items()}
            with torch.no_grad():
                proba[i:i+64] = torch.softmax(mb["model"](**enc).logits, -1).cpu().numpy()
    else:
        classes = list(ar_service._router.classes_)
        proba = ar_service._router.predict_proba(texts)     # linear soft-vote
    pred = np.array(classes)[proba.argmax(1)]
    top3_hit = np.mean([samp["cat"].values[i] in [classes[j] for j in np.argsort(proba[i])[::-1][:3]]
                        for i in range(len(texts))])
    y_true = np.array([classes.index(c) for c in samp["cat"].values])
    y_pred = np.array([classes.index(c) for c in pred])
    acc = float((y_true == y_pred).mean())
    ar_names = [ar_service.CANON_AR.get(c, c) for c in classes]
    print(f"[text] top1={acc:.4f} top3={top3_hit:.4f} in {time.time()-t0:.0f}s", flush=True)

    save_results("text", {
        "model": f"Arabic specialty router ({backend})",
        "modality": "clinical text (Arabic)", "task": "20-way specialty routing",
        "router_backend": backend,
        "dataset": "HELD-OUT test split (seed=0, 15%) of the >=200k training corpus — never trained on",
        "is_external_source": False,
        "new_source_note": "New external Arabic medical-QA with a mappable 20-specialty taxonomy was gated (HTTP 401); held-out split used instead.",
        "classes": classes, "class_names_ar": ar_names,
        "n_test": int(len(texts)), "accuracy": round(acc, 4), "top3_accuracy": round(float(top3_hit), 4),
    }, y_true=y_true, y_pred=y_pred)


if __name__ == "__main__":
    main()
