# -*- coding: utf-8 -*-
"""Assemble External_Validation.ipynb from results/*.{npz,json}.

Each model's cell loads its saved predictions (no re-inference), renders a confusion matrix,
prints accuracy + a per-class report, and the final cell ranks all models. Written with
nbformat; executed with nbconvert so the plots are baked in.
"""
import os, json, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
# project root (next to "Radiology Hub.html") — most discoverable place for the user
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "External_Validation.ipynb")

SETUP = r'''
import os, json, numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
plt.rcParams["figure.dpi"] = 110

RESULTS = os.path.join("api", "eval_external", "results")
if not os.path.isdir(RESULTS):
    RESULTS = os.path.join("eval_external", "results")

def load(name):
    d = np.load(os.path.join(RESULTS, f"{name}.npz"), allow_pickle=True)
    meta = json.load(open(os.path.join(RESULTS, f"{name}.json"), encoding="utf-8"))
    return d, meta

def show_confusion(name, title, source=None):
    d, meta = load(name)
    yt, yp = d["y_true"], d["y_pred"]
    if source is not None and "source" in d:
        m = d["source"] == source
        yt, yp = yt[m], yp[m]
    labels = meta.get("classes")   # English on axes (matplotlib lacks Arabic shaping)
    ncol = len(labels)
    cm = confusion_matrix(yt, yp, labels=list(range(ncol)))
    fig, ax = plt.subplots(figsize=(min(1.1*ncol+2, 12), min(1.0*ncol+2, 11)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(ncol)); ax.set_yticks(range(ncol))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thr = cm.max()/2 if cm.max() else 1
    for i in range(ncol):
        for j in range(ncol):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thr else "black")
    acc = accuracy_score(yt, yp)
    ax.set_title(f"{title}\nn={len(yt)}  accuracy={acc:.3f}", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04); plt.tight_layout(); plt.show()
    print("dataset:", meta["dataset"])
    print(classification_report(yt, yp, target_names=[str(l) for l in labels], zero_division=0))
    return meta
'''

RANK = r'''
rows = []
for name in ["brain", "pneumonia", "chest", "text"]:
    p = os.path.join(RESULTS, f"{name}.json")
    if not os.path.exists(p): continue
    m = json.load(open(p, encoding="utf-8"))
    headline = (m.get("mean_auc") if name == "chest" else m.get("accuracy"))
    metric = "mean ROC-AUC" if name == "chest" else "accuracy"
    rows.append((m["model"].split("(")[0].strip()[:34], name, m["n_test"], metric, headline,
                 "external" if m.get("is_external_source") else "held-out"))
rows.sort(key=lambda r: (r[4] is not None, r[4]), reverse=True)

print(f"{'#':<2}{'model':36}{'n':>6}  {'metric':13}{'score':>7}  source")
print("-"*78)
for i,(mdl,nm,n,metric,score,src) in enumerate(rows,1):
    s = f"{score:.3f}" if score is not None else "  -  "
    print(f"{i:<2}{mdl:36}{n:>6}  {metric:13}{s:>7}  {src}")

names=[r[0] for r in rows][::-1]; scores=[r[4] or 0 for r in rows][::-1]
colors=["#0ec2b8" if r[5]=="external" else "#f5a524" for r in rows][::-1]
fig,ax=plt.subplots(figsize=(9,3.2))
ax.barh(names,scores,color=colors)
for i,s in enumerate(scores): ax.text(s+0.01,i,f"{s:.3f}",va="center",fontsize=9)
ax.set_xlim(0,1.05); ax.set_xlabel("headline score (acc, or mean-AUC for chest)")
ax.set_title("External validation — models ranked"); plt.tight_layout(); plt.show()
'''

CHEST_AUC = r'''
d, meta = load("chest")
pl = meta["per_label_auc"]
items = sorted(pl.items(), key=lambda kv: kv[1]["auc"], reverse=True)
names=[k for k,_ in items]; aucs=[v["auc"] for _,v in items]   # English keys (axis-safe)
fig,ax=plt.subplots(figsize=(9,5))
ax.barh(names[::-1],aucs[::-1],color="#1f8ef1")
ax.axvline(0.5,color="#e5484d",ls="--",lw=1,label="chance (0.5)")
for i,a in enumerate(aucs[::-1]): ax.text(a+0.005,i,f"{a:.3f}",va="center",fontsize=8)
ax.set_xlim(0,1.0); ax.set_title(f"Chest — per-label ROC-AUC (mean {meta['mean_auc']}, n={meta['n_test']})")
ax.legend(); plt.tight_layout(); plt.show()
'''


def main():
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            "# التحقق الخارجي من النماذج — External Validation\n\n"
            "اختبار كل نموذج على بيانات **لم يتدرّب عليها** (≥2500 عيّنة حيثما أمكن)، مع "
            "مصفوفة الالتباس (confusion matrix)، الدقة، وترتيب النماذج.\n\n"
            "**الصدق المنهجي (مهم):**\n"
            "- **الدماغ**: كل مجموعات الدماغ ٤-أصناف العامة هي نفس الـ7023 صورة (احتكار مصدر). "
            "استخدمنا المجموعة المحجوزة الخالية من التسريب + صور PranomVignesh الجديدة فعلاً "
            "(dHash > 5 مقابل التدريب).\n"
            "- **الالتهاب الرئوي**: اختبار **عبر-التوزيع** — تدرّب على أطفال (Kermany) واختُبر على "
            "بالغين (NIH). الانخفاض عن ٩٦٪ هو انزياح توزيع حقيقي، مو خطأ بالنموذج.\n"
            "- **الصدر**: نموذج مُدرّب مسبقاً، اختُبر على شريحة جديدة ٢٥٠٠ من ChestMNIST (NIH).\n"
            "- **النص**: مصادر QA عربية جديدة كانت مقفلة (401)، فاستُخدمت المجموعة المحجوزة (غير مُدرّب عليها).\n"),
        new_code_cell(SETUP.strip()),
        new_markdown_cell(
            "## 🧠 الدماغ — Brain MRI (4 classes)\n\n"
            "مجموعة الاختبار = **699** محجوزة خالية من التسريب (نفس المصدر) + **1801** صورة "
            "PranomVignesh جديدة فعلاً (معاد تصديرها/مُعزّزة بـRoboflow). النتيجة المدمجة تخفي "
            "فرقاً كبيراً بين المصدرين — نعرضه صراحةً:"),
        new_code_cell(
            'd, meta = load("brain")\n'
            'print("accuracy by source:", json.dumps(meta["accuracy_by_source"], indent=2))\n'
            'print("\\nالمحجوزة (نفس المصدر) عالية؛ الصور المُعزّزة الجديدة تنخفض بوضوح.")'),
        new_markdown_cell("### المحجوزة (نفس المصدر) — held-out 699"),
        new_code_cell('show_confusion("brain", "Brain — leak-free held-out (same source)", source="heldout");'),
        new_markdown_cell(
            "### الجديدة المُعزّزة — PranomVignesh novel 1801\n"
            "الانخفاض يتركّز في صنف واحد: الورم النخامي (pituitary) ~34٪ يُقرأ «لا ورم». "
            "أما «لا ورم» نفسه فدقته 0.99 — أي النموذج يعرف «لا ورم» جيداً لكنه يخلط pituitary بها."),
        new_code_cell('show_confusion("brain", "Brain — PranomVignesh novel", source="prano_novel");'),
        new_markdown_cell(
            "### محاولة الإصلاح (v3) + التشخيص — attempted fix & diagnosis\n"
            "جرّبنا **إصلاحاً**: إعادة تدريب بتعزيز أقوى (تدوير ٢٥°، قص عشوائي، ضبابية، مسح عشوائي) = "
            "`train_brain_v3.py`. النتيجة **دحضت الفرضية**: v3 لم يُحسّن الصور الجديدة بل أسوأ. "
            "والدليل الحاسم أدناه أن المشكلة في **بيانات PranomVignesh** لصنف pituitary وليست ضعفاً عاماً "
            "بالنموذج: pituitary يفشل حتى على صور PranomVignesh المطابقة تقريباً للتدريب.\n\n"
            "**الخلاصة الصادقة:** الرقم الموثوق للدماغ هو المجموعة المحجوزة الخالية من التسريب = "
            "**98.9٪** (v2). النموذج المخدوم يبقى v2."),
        new_code_cell(
            'v2 = json.load(open(os.path.join(RESULTS,"brain.json"),encoding="utf-8"))\n'
            'v3p = os.path.join(RESULTS,"brain_v3.json")\n'
            'print("brain v2  by source:", json.dumps(v2["accuracy_by_source"]))\n'
            'if os.path.exists(v3p):\n'
            '    v3 = json.load(open(v3p,encoding="utf-8"))\n'
            '    print("brain v3  by source:", json.dumps(v3["accuracy_by_source"]))\n'
            '    print("\\n-> stronger augmentation did NOT help the novel set (hypothesis refuted); '
            'held-out stays ~0.98-0.99. v2 remains the served model.")'),
        new_markdown_cell("## 🦠 الالتهاب الرئوي — Pneumonia (cross-distribution: NIH adults)"),
        new_code_cell('m = show_confusion("pneumonia", "Pneumonia — NIH (cross-distribution)")\n'
                      'print("in-domain (Kermany) was acc 0.963 / spec 0.915 — the gap is domain shift")'),
        new_markdown_cell("## 🫁 الصدر — Chest 14-label (per-label AUC is the real metric)"),
        new_code_cell(CHEST_AUC.strip()),
        new_code_cell('show_confusion("chest", "Chest — any-finding vs no-finding (illustrative)");'),
        new_markdown_cell("## 📝 النص العربي — Arabic specialty router (20 classes, held-out)"),
        new_code_cell('show_confusion("text", "Arabic router — 20 specialties");'),
        new_markdown_cell("## 🏆 الترتيب النهائي — Ranking\n"
                          "أخضر = مصدر خارجي مستقل · أصفر = مجموعة محجوزة"),
        new_code_cell(RANK.strip()),
    ]
    nb["cells"] = cells
    nbf.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
