# -*- coding: utf-8 -*-
"""
GPU / Colab — fine-tune an Arabic transformer (MARBERT) for the specialty router.
This is the "plug-in when GPU is available" upgrade to the CPU linear router.

WHY: the linear router is at the data ceiling — every bag-of-words classifier we swept
(SGD/LinearSVC/LogReg/ComplementNB) lands in 0.65-0.71 top-1. A transformer is the only
realistic way past it. `ar_service.py` really does auto-detect `models/ar/marbert_router/`
and route with it, falling back to the linear model if it's absent or unloadable.

RUN LOCALLY ON A SMALL GPU (e.g. GTX 1650, 4 GB):
  pip install transformers datasets accelerate
  python ar_train_transformer.py
  Defaults below are tuned for 4 GB: batch 16 x grad-accum 2 (= effective 32), maxlen 128, fp16.
  Turing (sm_75) has no tensor cores on the 1650, so fp16 mainly buys MEMORY, not much speed —
  expect hours, not minutes. Drop BATCH to 8 if you still hit CUDA OOM.

RUN ON COLAB (T4/A100 — much faster):
  1) Runtime -> Change runtime type -> GPU.
  2) Upload data/arabic/router_data.parquet (from ar_data_prep.py), or mount Drive and set ROUTER_PARQUET.
  3) !pip install -q transformers datasets accelerate scikit-learn pandas pyarrow
  4) !python ar_train_transformer.py   (BATCH=32 is fine there)
  5) Download marbert_router/ -> place under api/models/ar/marbert_router/ .

Model: UBC-NLP/MARBERT (Arabic BERT trained on ~1B Arabic tweets + MSA) — strong for dialectal
clinical text. AraBERT (aubmindlab/bert-base-arabertv2) is an alternative.
"""
import os
import json
import numpy as np
import pandas as pd

MODEL_NAME = os.environ.get("AR_TRANSFORMER", "UBC-NLP/MARBERT")
ROUTER_PARQUET = os.environ.get("ROUTER_PARQUET", "data/arabic/router_data.parquet")
OUT_DIR = os.environ.get("OUT_DIR", "models/ar/marbert_router")
EPOCHS = int(os.environ.get("EPOCHS", "3"))
MAXLEN = int(os.environ.get("MAXLEN", "128"))
BATCH = int(os.environ.get("BATCH", "16"))          # 4 GB-safe; Colab can raise to 32
ACCUM = int(os.environ.get("ACCUM", "2"))           # effective batch = BATCH * ACCUM
SAMPLE = int(os.environ.get("AR_SAMPLE", "0"))      # >0 = quick smoke run on a subset


def main():
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              TrainingArguments, Trainer, DataCollatorWithPadding)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to torch.\n"
                         "  - Check `python -c \"import torch; print(torch.version.cuda)\"` is not None.\n"
                         "  - A CPU-only torch build will never see the GPU; install a CUDA build.\n"
                         "  - Fine-tuning 166k examples on CPU is not viable, so this script stops here.")
    print(f"[gpu] {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    df = pd.read_parquet(ROUTER_PARQUET).dropna(subset=["text", "category"])
    df = df[df["text"].str.len() >= 15]
    if SAMPLE:
        df = df.groupby("category", group_keys=False).apply(
            lambda g: g.sample(min(len(g), max(SAMPLE // 20, 20)), random_state=0))
    labels = sorted(df["category"].unique())
    l2i = {c: i for i, c in enumerate(labels)}
    df["label"] = df["category"].map(l2i)

    # SAME split protocol as the linear router (seed=0, 15% stratified) -> directly comparable
    tr, te = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=0)
    print(f"[data] train={len(tr)} test={len(te)} classes={len(labels)} "
          f"| batch={BATCH} x accum={ACCUM} (effective {BATCH*ACCUM}) maxlen={MAXLEN}")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def enc(batch):
        return tok(batch["text"], truncation=True, max_length=MAXLEN)
    ds_tr = Dataset.from_pandas(tr[["text", "label"]]).map(enc, batched=True)
    ds_te = Dataset.from_pandas(te[["text", "label"]]).map(enc, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels),
        id2label={i: c for c, i in l2i.items()}, label2id=l2i)

    def metrics(eval_pred):
        logits, y = eval_pred
        p = logits.argmax(-1)
        return {"accuracy": accuracy_score(y, p),
                "macro_f1": f1_score(y, p, average="macro"),
                "top3": top_k_accuracy_score(y, logits, k=3, labels=list(range(len(labels))))}

    args = TrainingArguments(
        output_dir=OUT_DIR, num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,  # 4 GB disk is tight
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.06, logging_steps=200,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        fp16=True,                      # on a 1650 this saves MEMORY (no tensor cores for speed)
        gradient_checkpointing=True,    # trades compute for VRAM so BERT-base fits in 4 GB
        dataloader_num_workers=2, report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te,
                      tokenizer=tok, data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()
    res = trainer.evaluate()
    print("[MARBERT eval]", json.dumps({k: round(float(v), 4) for k, v in res.items() if isinstance(v, (int, float))}))
    trainer.save_model(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    json.dump({"labels": labels, "model_name": MODEL_NAME, "eval": {k: float(v) for k, v in res.items() if isinstance(v, (int, float))}},
              open(os.path.join(OUT_DIR, "router_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("MARBERT_ROUTER_SAVED ->", OUT_DIR)


if __name__ == "__main__":
    main()
