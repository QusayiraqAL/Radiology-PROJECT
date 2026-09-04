# -*- coding: utf-8 -*-
"""Generate Pneumonia_Colab_Test.ipynb — load the trained pneumonia_v2.pt in Colab and test it
on the PneumoniaMNIST held-out TEST split (downloaded in-notebook), with metrics + confusion
matrix + example predictions, plus an optional 'upload your own X-ray' cell. Self-contained:
rebuilds the architecture inline (no project files needed besides the uploaded checkpoint)."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                   "Pneumonia_Colab_Test.ipynb")

INSTALL = "!pip -q install torch torchvision medmnist scikit-learn matplotlib"

UPLOAD = r'''
# ارفع ملف النموذج: api/models/pneumonia_v2.pt  (~43 MB)
import os
if not os.path.exists("pneumonia_v2.pt"):
    from google.colab import files
    up = files.upload()                      # اختر pneumonia_v2.pt
    name = next(iter(up))
    if name != "pneumonia_v2.pt":
        os.rename(name, "pneumonia_v2.pt")
print("model file:", os.path.getsize("pneumonia_v2.pt")/1e6, "MB")
'''

LOAD = r'''
import torch, torch.nn as nn, torchvision, numpy as np

ck = torch.load("pneumonia_v2.pt", map_location="cpu", weights_only=False)
SIZE   = ck.get("size", 224)
THRESH = ck.get("threshold", 0.5)           # عتبة القرار المضبوطة على التحقق (Youden's J)
DROP   = ck.get("dropout", 0.3)
MEAN   = ck.get("mean", [0.485, 0.456, 0.406])
STD    = ck.get("std",  [0.229, 0.224, 0.225])

# نفس معمارية التدريب: ResNet-18 + رأس Dropout->Linear(512,2)
net = torchvision.models.resnet18(weights=None)
net.fc = nn.Sequential(nn.Dropout(DROP), nn.Linear(net.fc.in_features, 2))
net.load_state_dict(ck["state_dict"])
net.eval()
DEV = "cuda" if torch.cuda.is_available() else "cpu"; net.to(DEV)
_M = torch.tensor(MEAN).view(3,1,1); _S = torch.tensor(STD).view(3,1,1)

def preprocess(pil):
    im = pil.convert("RGB").resize((SIZE, SIZE))
    x = torch.from_numpy(np.asarray(im, np.float32)/255.0).permute(2,0,1)
    return (x - _M) / _S

@torch.no_grad()
def predict_proba(pils, bs=64):
    out = []
    for i in range(0, len(pils), bs):
        xb = torch.stack([preprocess(p) for p in pils[i:i+bs]]).to(DEV)
        out.append(torch.softmax(net(xb), 1)[:,1].cpu().numpy())   # احتمال الالتهاب
    return np.concatenate(out)

print(f"loaded ResNet-18 | size={SIZE} threshold={THRESH:.2f} device={DEV}")
'''

DATA = r'''
# تحميل مجموعة الاختبار الرسمية (لم تُستخدم في التدريب): PneumoniaMNIST-224
import medmnist
from medmnist import INFO
from PIL import Image
DataClass = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])
ds = DataClass(split="test", download=True, size=224)
imgs = ds.imgs                       # (624, 224, 224) رمادية
labels = ds.labels.reshape(-1)       # 0=طبيعي، 1=التهاب رئوي
pils = [Image.fromarray(a) for a in imgs]
print("test images:", len(pils), "| normal:", int((labels==0).sum()), "pneumonia:", int((labels==1).sum()))
'''

EVAL = r'''
from sklearn.metrics import (accuracy_score, roc_auc_score, confusion_matrix,
                             recall_score, precision_score, classification_report)
import matplotlib.pyplot as plt

prob = predict_proba(pils)
pred = (prob >= THRESH).astype(int)

acc  = accuracy_score(labels, pred)
auc  = roc_auc_score(labels, prob)
tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0,1]).ravel()
sens = tp/(tp+fn); spec = tn/(tn+fp)
print(f"accuracy    = {acc:.4f}")
print(f"AUC         = {auc:.4f}")
print(f"sensitivity = {sens:.4f}  (كشف الالتهاب)")
print(f"specificity = {spec:.4f}  (استبعاد الطبيعي)")
print("\n", classification_report(labels, pred, target_names=["normal","pneumonia"], zero_division=0))

cm = confusion_matrix(labels, pred, labels=[0,1])
fig, ax = plt.subplots(figsize=(4.5,4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["normal","pneumonia"]); ax.set_yticklabels(["normal","pneumonia"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(2):
    for j in range(2):
        ax.text(j,i,int(cm[i,j]),ha="center",va="center",
                color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=13)
ax.set_title(f"Pneumonia — test (n={len(labels)})  acc={acc:.3f}")
plt.tight_layout(); plt.show()
'''

EXAMPLES = r'''
# عيّنات من الاختبار مع التنبؤ (أخضر=صحيح، أحمر=خطأ)
import numpy as np, matplotlib.pyplot as plt
rng = np.random.RandomState(0)
idx = rng.choice(len(pils), 12, replace=False)
fig, axes = plt.subplots(3,4, figsize=(11,8))
for ax, i in zip(axes.ravel(), idx):
    ax.imshow(imgs[i], cmap="gray"); ax.axis("off")
    p = prob[i]; yhat = int(p>=THRESH); y = int(labels[i])
    ok = (yhat==y)
    ax.set_title(f"true={'PNEU' if y else 'NORM'}\npred={'PNEU' if yhat else 'NORM'} ({p*100:.0f}%)",
                 color="green" if ok else "red", fontsize=10)
plt.tight_layout(); plt.show()
'''

OWN = r'''
# (اختياري) جرّب صورة أشعة صدر من عندك
from google.colab import files
from PIL import Image
up = files.upload()
for name in up:
    im = Image.open(name)
    p = float(predict_proba([im])[0])
    verdict = "التهاب رئوي مشتبه به" if p>=THRESH else "طبيعي — لا مؤشر التهاب"
    print(f"{name}: P(pneumonia)={p*100:.1f}%  ->  {verdict}  (عتبة {THRESH*100:.0f}%)")
'''


def main():
    nb = new_notebook()
    nb["cells"] = [
        new_markdown_cell(
            "# تجربة كاشف الالتهاب الرئوي — Pneumonia Detector (Colab)\n\n"
            "يحمّل النموذج المدرّب (`pneumonia_v2.pt`) ويختبره على **مجموعة الاختبار الرسمية** "
            "لـPneumoniaMNIST (٦٢٤ صورة لم تُستخدم في التدريب)، مع الدقة/الحساسية/النوعية "
            "ومصفوفة الالتباس وأمثلة، بالإضافة لخلية لرفع صورتك.\n\n"
            "**الخطوات:** شغّل الخلايا بالترتيب. عند خلية الرفع، ارفع الملف من جهازك:\n"
            "`api/models/pneumonia_v2.pt` (~43 MB).\n\n"
            "**المتوقّع** (النتيجة المقاسة عندنا على نفس المجموعة): دقة ~**0.96**، "
            "حساسية ~**0.99**، نوعية ~**0.91**، AUC ~**0.99**. "
            "GPU غير مطلوب — يشتغل على CPU خلال دقائق."),
        new_markdown_cell("## 1) تثبيت المكتبات"),
        new_code_cell(INSTALL),
        new_markdown_cell("## 2) رفع النموذج (pneumonia_v2.pt)"),
        new_code_cell(UPLOAD.strip()),
        new_markdown_cell("## 3) تحميل النموذج (نفس معمارية ومعالجة الخادم)"),
        new_code_cell(LOAD.strip()),
        new_markdown_cell("## 4) سحب بيانات الاختبار (PneumoniaMNIST test)"),
        new_code_cell(DATA.strip()),
        new_markdown_cell("## 5) التقييم: دقة / حساسية / نوعية / AUC + مصفوفة الالتباس"),
        new_code_cell(EVAL.strip()),
        new_markdown_cell("## 6) أمثلة من الاختبار مع التنبؤ"),
        new_code_cell(EXAMPLES.strip()),
        new_markdown_cell("## 7) (اختياري) جرّب صورتك"),
        new_code_cell(OWN.strip()),
    ]
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
