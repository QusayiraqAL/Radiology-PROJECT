# -*- coding: utf-8 -*-
"""Generate MARBERT_Colab_Train.ipynb — a self-contained Colab notebook that fine-tunes
MARBERT for the 20-way Arabic specialty router, on the user's exact data + split, and
packages the result so api/ar_service.py auto-loads it (router_backend -> 'marbert')."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "MARBERT_Colab_Train.ipynb")

# pin <5: transformers 5.x removed the Trainer `tokenizer=` arg this notebook uses
INSTALL = "!pip -q install \"transformers>=4.44,<5\" datasets accelerate scikit-learn pandas pyarrow"

GPU_CHECK = r'''
import torch
assert torch.cuda.is_available(), "Runtime -> Change runtime type -> GPU (T4) ثم أعد التشغيل"
print("GPU:", torch.cuda.get_device_name(0),
      f"| {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
'''

UPLOAD = r'''
# ارفع ملف router_data.parquet من جهازك:
#   api/data/arabic/router_data.parquet   (حجمه ~41 MB)
import os
if not os.path.exists("router_data.parquet"):
    from google.colab import files
    up = files.upload()          # اختر router_data.parquet
    name = next(iter(up))
    if name != "router_data.parquet":
        os.rename(name, "router_data.parquet")
print("data ready:", os.path.getsize("router_data.parquet")/1e6, "MB")
'''

DATA = r'''
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

SEED = 0
df = pd.read_parquet("router_data.parquet").dropna(subset=["text", "category"])
df = df[df["text"].str.len() >= 15].reset_index(drop=True)
labels = sorted(df["category"].unique())          # نفس ترتيب الأصناف مثل النموذج الخطي
l2i = {c: i for i, c in enumerate(labels)}
df["label"] = df["category"].map(l2i)

# نفس التقسيم تماماً مثل النموذج الخطي (seed=0, 15% محجوزة) -> مقارنة عادلة
tr_df, te_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=SEED)
print(f"train={len(tr_df)}  test(held-out)={len(te_df)}  classes={len(labels)}")
print("classes:", labels)
'''

TOKENIZE = r'''
from transformers import AutoTokenizer
from datasets import Dataset

MODEL_NAME = "UBC-NLP/MARBERT"     # BERT عربي مدرّب على ~1B تغريدة + فصحى
MAXLEN = 128
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def enc(b): return tok(b["text"], truncation=True, max_length=MAXLEN)
ds_tr = Dataset.from_pandas(tr_df[["text","label"]]).map(enc, batched=True)
ds_te = Dataset.from_pandas(te_df[["text","label"]]).map(enc, batched=True)
'''

TRAIN = r'''
import numpy as np
from transformers import (AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(labels),
    id2label={i:c for c,i in l2i.items()}, label2id=l2i)

def metrics(ep):
    logits, y = ep
    p = logits.argmax(-1)
    return {"accuracy": accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro"),
            "top3": top_k_accuracy_score(y, logits, k=3, labels=list(range(len(labels))))}

args = TrainingArguments(
    output_dir="out", num_train_epochs=3,
    per_device_train_batch_size=32, per_device_eval_batch_size=64,
    learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.06,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
    load_best_model_at_end=True, metric_for_best_model="accuracy",
    fp16=True, logging_steps=200, report_to=[])

trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te,
                  tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
                  compute_metrics=metrics)
trainer.train()
res = trainer.evaluate()
print("HELD-OUT:", {k: round(float(v),4) for k,v in res.items() if isinstance(v,(int,float))})
'''

EVAL = r'''
# فجوة التعميم: دقة على عيّنة من التدريب مقابل الاختبار المحجوز
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

te_pred = trainer.predict(ds_te)
te_logits, y_te = te_pred.predictions, te_pred.label_ids
y_hat = te_logits.argmax(-1)
test_acc = (y_hat == y_te).mean()
top3 = top_k_accuracy_score(y_te, te_logits, k=3, labels=list(range(len(labels))))

samp = tr_df.sample(min(5000, len(tr_df)), random_state=0)
ds_s = Dataset.from_pandas(samp[["text","label"]]).map(enc, batched=True)
tr_logits = trainer.predict(ds_s).predictions
train_acc = (tr_logits.argmax(-1) == samp["label"].values).mean()

print(f"top-1 (test)   = {test_acc:.4f}")
print(f"top-3 (test)   = {top3:.4f}")
print(f"train accuracy = {train_acc:.4f}")
print(f"train/test gap = {train_acc - test_acc:+.4f}")
print("\n", classification_report(y_te, y_hat, target_names=labels, zero_division=0))

cm = confusion_matrix(y_te, y_hat, labels=range(len(labels)))
fig, ax = plt.subplots(figsize=(11,10))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
thr = cm.max()/2
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j,i,int(cm[i,j]),ha="center",va="center",fontsize=7,
                color="white" if cm[i,j]>thr else "black")
ax.set_title(f"MARBERT router — held-out (n={len(y_te)})  top-1={test_acc:.3f}  top-3={top3:.3f}")
fig.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout(); plt.show()
'''

SAVE = r'''
# احفظ النموذج بالشكل الذي يتوقّعه api/ar_service.py (يكتشفه تلقائياً)
import json, shutil, os
OUT = "marbert_router"
trainer.save_model(OUT); tok.save_pretrained(OUT)
json.dump({"labels": labels, "model_name": MODEL_NAME,
           "eval": {"top1": float(test_acc), "top3": float(top3),
                    "train_acc": float(train_acc), "gap": float(train_acc-test_acc)}},
          open(os.path.join(OUT,"router_meta.json"),"w",encoding="utf-8"),
          ensure_ascii=False, indent=2)
shutil.make_archive("marbert_router","zip",OUT)
print("files:", os.listdir(OUT))
from google.colab import files
files.download("marbert_router.zip")
'''


def main():
    nb = new_notebook()
    nb["cells"] = [
        new_markdown_cell(
            "# تدريب MARBERT للتوجيه النصّي — MARBERT Router (Colab GPU)\n\n"
            "يرفع دقة توجيه التخصص (20 صنف) لأقصى حد ممكن عبر محوّل عربي، بدل النموذج الخطي "
            "(الذي سقفه ~70٪ top-1).\n\n"
            "**الخطوات:**\n"
            "1. `Runtime → Change runtime type → GPU` (T4 كافٍ).\n"
            "2. شغّل الخلايا بالترتيب. عند خلية الرفع، ارفع "
            "`api/data/arabic/router_data.parquet` من جهازك.\n"
            "3. في النهاية يتنزّل `marbert_router.zip`.\n\n"
            "**التوقّع الصادق:** MARBERT عادةً يعطي top-1 بحدود **78–84٪** (صعوداً من 70.5٪) "
            "و top-3 **~94–96٪**. الوصول 95٪ top-1 غير مرجّح لأن التخصصات الـ20 متداخلة فعلاً "
            "— لكن هذا أقصى حدّ واقعي، وأفضل بوضوح من الخطي.\n\n"
            "**الوقت:** ~30–50 دقيقة على T4 (3 حِقب × 166 ألف مثال)."),
        new_markdown_cell("## 1) تثبيت المكتبات"),
        new_code_cell(INSTALL),
        new_markdown_cell("## 2) التأكد من الـ GPU"),
        new_code_cell(GPU_CHECK.strip()),
        new_markdown_cell("## 3) رفع البيانات (router_data.parquet)"),
        new_code_cell(UPLOAD.strip()),
        new_markdown_cell("## 4) تحميل + تقسيم (نفس بذرة النموذج الخطي)"),
        new_code_cell(DATA.strip()),
        new_markdown_cell("## 5) الترميز (MARBERT tokenizer)"),
        new_code_cell(TOKENIZE.strip()),
        new_markdown_cell("## 6) التدريب"),
        new_code_cell(TRAIN.strip()),
        new_markdown_cell("## 7) التقييم: top-1 / top-3 / فجوة التعميم + مصفوفة الالتباس"),
        new_code_cell(EVAL.strip()),
        new_markdown_cell("## 8) الحفظ والتنزيل (بالشكل الذي يقرأه الخادم)"),
        new_code_cell(SAVE.strip()),
        new_markdown_cell(
            "## 9) التركيب في المشروع\n\n"
            "بعد تنزيل `marbert_router.zip`:\n"
            "1. فك الضغط داخل: `api/models/ar/marbert_router/` "
            "(يجب أن يحتوي `config.json`, `model.safetensors`, ملفات الـtokenizer, و`router_meta.json`).\n"
            "2. ثبّت مكتبات المحوّل في بيئة المشروع مرّة واحدة:\n"
            "   `api/venv/Scripts/python -m pip install transformers torch`\n"
            "3. أعد تشغيل الخادم (`start_server.bat`).\n\n"
            "`ar_service.py` يكتشف المجلد تلقائياً ويستخدم MARBERT للتوجيه، ويرجع للنموذج الخطي "
            "إن غاب. تحقّق عبر `GET /models` أن `router_backend` صار **`marbert`**.\n\n"
            "> ملاحظة: مقاس MARBERT ~500MB. تأكد من وجود مساحة قرص كافية قبل التركيب."),
    ]
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
