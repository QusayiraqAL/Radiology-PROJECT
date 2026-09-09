# -*- coding: utf-8 -*-
"""
AI Radiology Hub — Real Multi-Model Prediction API
==================================================
Serves FOUR real, trained/validated systems:

  1) chest      — TorchXRayVision DenseNet121 (18 chest-X-ray pathologies), pretrained.
  2) pneumonia  — SmallXRayCNN trained on PneumoniaMNIST, held-out validated.
  3) brain      — ResNet-18 fine-tuned on real brain-MRI slices, held-out validated.
  4) symptoms_ar— Arabic symptom text -> specialty (20) -> diagnosis, plus an input filter.
                  Trained on >=200k real Arabic medical entries (see ar_service.py).

Every /models entry carries the REAL measured metrics from models/*.

Run:  uvicorn main:app --host 127.0.0.1 --port 8000
"""
import io
import os
import json
import math
import time

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image

from nets import (SmallXRayCNN, build_brain_resnet, build_pneumonia_resnet,
                  build_medmnist_backbone)
from img_utils import crop_brain_region
import preproc
from gradcam import gradcam_overlay   # educational "where does the model look" heatmap

HERE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(HERE, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="AI Radiology Hub API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

REGISTRY = {}   # id -> dict(meta, predictor)


def _finite_only(o):
    """Replace NaN/Infinity with None, recursively.

    json.dump writes a bare `NaN` token for a non-finite float and json.load reads it back
    without complaint, so a metrics file can carry one for months in silence. Starlette then
    serializes every response with allow_nan=False and the whole endpoint dies with
    "ValueError: Out of range float values are not JSON compliant: nan" - a 500 on /models
    that names neither the file nor the field. derma_v2 and derma_bin both shipped a NaN
    mean_confidence this way (their fp16 eval pass produced non-finite probabilities; the
    same array is why both recorded a null test_auc, later recovered in fp32 on CPU).

    Sanitising at the single point where metrics enter the process is what makes that a
    missing value in one card instead of an outage: no training script, present or future,
    can take the API down by writing a number sklearn or numpy handed it.
    """
    if isinstance(o, dict):
        return {k: _finite_only(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_finite_only(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _load_metrics(name):
    p = os.path.join(MODEL_DIR, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return _finite_only(json.load(f))
    return None


def assess_ood(probs):
    """Heuristic out-of-distribution / low-confidence check on a softmax vector.

    Not a guarantee (neural nets can be over-confident on OOD) — a teaching signal that the
    image may be the wrong modality or an unusual case. Flags when the model isn't clearly
    committing to any class (low max prob or near-uniform distribution).
    """
    p = np.asarray(probs, dtype=np.float64)
    n = len(p)
    max_p = float(p.max())
    ent = float(-(p * np.log(p + 1e-9)).sum() / np.log(n)) if n > 1 else 0.0
    suspect = (max_p < 0.60) if n <= 2 else (max_p < 0.45 or ent > 0.85)
    return {"suspect": bool(suspect), "max_prob": round(max_p, 3), "norm_entropy": round(ent, 3),
            "message_ar": ("الصورة قد لا تناسب هذا النموذج (نوع فحص مختلف) أو الحالة غير اعتيادية — "
                           "الثقة منخفضة، فسّر النتيجة بحذر." if suspect else None)}


# ---------------------------------------------------------------------------
# 1) CHEST — TorchXRayVision DenseNet121 (pretrained, real)
# ---------------------------------------------------------------------------
PATHOLOGY_AR = {
    "Atelectasis": "انخماص رئوي", "Consolidation": "تصلّد رئوي", "Infiltration": "ارتشاح رئوي",
    "Pneumothorax": "استرواح صدري", "Edema": "وذمة رئوية", "Emphysema": "نفاخ رئوي",
    "Fibrosis": "تليف رئوي", "Effusion": "انصباب جنبي", "Pneumonia": "التهاب رئوي",
    "Pleural_Thickening": "تسمّك جنبي", "Cardiomegaly": "تضخم القلب", "Nodule": "عقيدة رئوية",
    "Mass": "كتلة", "Hernia": "فتق حجابي", "Lung Lesion": "آفة رئوية", "Fracture": "كسر",
    "Lung Opacity": "عتامة رئوية", "Enlarged Cardiomediastinum": "توسع المنصف القلبي",
}

print(f"[*] Loading chest model (TorchXRayVision) on {DEVICE} ...")
_chest_model = xrv.models.DenseNet(weights="densenet121-res224-all").to(DEVICE).eval()
_chest_tf = torchvision.transforms.Compose([
    xrv.datasets.XRayCenterCrop(),
    xrv.datasets.XRayResizer(224),
])


def predict_chest(pil_rgb):
    # torchxrayvision's pipeline wants a 2-D single-channel array. _read_image now hands
    # every predictor RGB (see its docstring), so the reduction happens here, where the
    # requirement actually lives, instead of being forced on all fourteen models.
    img = np.asarray(pil_rgb.convert("L"), dtype=np.float32)
    img = xrv.datasets.normalize(img, 255)
    img = img[None, ...]
    img = _chest_tf(img)
    tensor = torch.from_numpy(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _chest_model(tensor)[0].cpu().numpy()
    findings = []
    for name, prob in zip(_chest_model.pathologies, out):
        if not name:
            continue
        p = round(float(prob) * 100, 1)
        findings.append({
            "id": name, "name_en": name.replace("_", " "),
            "name_ar": PATHOLOGY_AR.get(name, name),
            "probability": p, "positive": p >= 50,
            "verdict": "above" if p >= 50 else ("near" if p >= 35 else "low"),
        })
    findings.sort(key=lambda f: f["probability"], reverse=True)
    return {
        "type": "multilabel", "threshold": 50, "calibration": "op_norm",
        "positives_count": sum(1 for f in findings if f["positive"]),
        "findings": findings,
    }


REGISTRY["chest"] = {
    "meta": {
        "id": "chest", "title_ar": "أشعة الصدر السينية — تحليل شامل",
        "modality": "chest X-ray", "title_en": "Chest X-ray multi-pathology",
        "pathologies": 18, "kind": "pretrained",
        "source": "TorchXRayVision DenseNet121 (densenet121-res224-all)",
        "edu": "قراءة منهجية لأشعة الصدر: 18 حالة (عتامات، انصباب، استرواح، تضخم قلب...). درّب عينك على مقاربة ABCDE — المجرى الهوائي، التنفّس/الرئتان، القلب، الحجاب، والعظام.",
        "metrics": _load_metrics("chest_metrics.json"),
    },
    "predict": predict_chest,
}


# ---------------------------------------------------------------------------
# 2) PNEUMONIA — prefers the v2 ResNet-18 @224 (transfer learning + augmentation +
#    dropout + label smoothing). Falls back to the v1 SmallXRayCNN @64 if v2 is absent.
# ---------------------------------------------------------------------------
_pneu = None
_pneu_arch = None          # "v2_resnet18" | "v1_smallcnn"
_IM_MEAN_T = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IM_STD_T = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_pneu_v2_path = os.path.join(MODEL_DIR, "pneumonia_v2.pt")
_pneu_v1_path = os.path.join(MODEL_DIR, "pneumonia_xray.pt")

if os.path.exists(_pneu_v2_path):
    print("[*] Loading pneumonia model (v2 ResNet-18 @224) ...")
    ckpt = torch.load(_pneu_v2_path, map_location=DEVICE, weights_only=False)
    _pneu = build_pneumonia_resnet(num_classes=2, pretrained=False,
                                   dropout=ckpt.get("dropout", 0.3)).to(DEVICE).eval()
    _pneu.load_state_dict(ckpt["state_dict"])
    _pneu_size = ckpt.get("size", 224)
    _pneu_th = ckpt.get("threshold", 0.5)
    _pneu_arch = "v2_resnet18"
    _pneu_metrics = _load_metrics("pneumonia_v2_metrics.json")
    _pneu_source = "ResNet-18 (ImageNet transfer) trained on PneumoniaMNIST-224 — real Kermany chest X-rays"
elif os.path.exists(_pneu_v1_path):
    print("[*] Loading pneumonia model (v1 SmallXRayCNN @64) ...")
    ckpt = torch.load(_pneu_v1_path, map_location=DEVICE, weights_only=False)
    _pneu = SmallXRayCNN(num_classes=1, in_ch=1).to(DEVICE).eval()
    _pneu.load_state_dict(ckpt["state_dict"])
    _pneu_size = ckpt.get("size", 64)
    _pneu_th = ckpt.get("threshold", 0.5)
    _pneu_arch = "v1_smallcnn"
    _pneu_metrics = _load_metrics("pneumonia_metrics.json")
    _pneu_source = "SmallXRayCNN trained on PneumoniaMNIST"
else:
    _pneu_metrics = None
    _pneu_source = "not trained yet"


def predict_pneumonia(pil_gray):
    if _pneu is None:
        raise HTTPException(503, "نموذج الالتهاب الرئوي غير محمّل — درّب النموذج أولاً")
    heatmap = None
    if _pneu_arch == "v2_resnet18":
        # same preprocessing as train_pneumonia_v2.eval_tf: RGB, 224, ImageNet normalize
        im = pil_gray.convert("RGB").resize((_pneu_size, _pneu_size))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - _IM_MEAN_T) / _IM_STD_T).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            prob = torch.softmax(_pneu(x), 1)[0, 1].item()
        heatmap = gradcam_overlay(_pneu, x, 1, im)     # highlight the pneumonia evidence
    else:
        # v1 is SmallXRayCNN(in_ch=1): it takes one channel, so reduce here rather than
        # upstream. Chest films are greyscale anyway, so nothing is lost by this conversion -
        # unlike the colour models, which is why the upstream one had to go.
        im = pil_gray.convert("L").resize((_pneu_size, _pneu_size))
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(x)[None, None, :, :].to(DEVICE)
        with torch.no_grad():
            prob = torch.sigmoid(_pneu(x).squeeze()).item()
    p = round(prob * 100, 1)
    positive = prob >= _pneu_th
    return {
        "type": "binary",
        "decision_threshold_pct": round(_pneu_th * 100, 1),
        "prediction_ar": "التهاب رئوي مشتبه به" if positive else "طبيعي — لا مؤشر التهاب",
        "prediction_en": "Pneumonia" if positive else "Normal",
        "positive": bool(positive),
        "findings": [
            {"id": "Pneumonia", "name_ar": "احتمالية الالتهاب الرئوي",
             "name_en": "Pneumonia probability", "probability": p,
             "positive": bool(positive),
             "verdict": "above" if positive else "low"},
        ],
        "ood": assess_ood([1 - prob, prob]),
        "heatmap": heatmap,
    }


REGISTRY["pneumonia"] = {
    "meta": {
        "id": "pneumonia", "title_ar": "كشف الالتهاب الرئوي (أشعة صدر)",
        "title_en": "Pneumonia detection", "modality": "chest X-ray",
        "kind": "trained", "available": _pneu is not None,
        "arch": _pneu_arch,
        "source": _pneu_source,
        "edu": "الالتهاب الرئوي على أشعة الصدر يظهر كعتامة/تصلّد بؤري. النموذج معاير للحساسية العالية (يمسك أغلب الحالات) — راقب الخريطة الحرارية: هل يشير لمنطقة التصلّد فعلاً؟",
        "metrics": _pneu_metrics,
    },
    "predict": predict_pneumonia,
}


# ---------------------------------------------------------------------------
# 3) BRAIN — ResNet-18 fine-tuned on real brain MRI (real, validated)
# ---------------------------------------------------------------------------
_brain = None
_brain_cropped = False     # v2 isolates the brain region before inference
BRAIN_AR = {"glioma": "ورم دبقي (Glioma)", "meningioma": "ورم سحائي (Meningioma)",
            "pituitary": "ورم الغدة النخامية (Pituitary)", "no-tumor": "لا يوجد ورم",
            "notumor": "لا يوجد ورم"}
_brain_v2_path = os.path.join(MODEL_DIR, "brain_tumor_mri_v2.pt")
_brain_v1_path = os.path.join(MODEL_DIR, "brain_tumor_mri.pt")
_brain_ckpt_path = _brain_v2_path if os.path.exists(_brain_v2_path) else _brain_v1_path
_is_brain_v2 = os.path.exists(_brain_v2_path)
_brain_metrics = _load_metrics("brain_v2_metrics.json" if _is_brain_v2 else "brain_metrics.json")
_brain_source = ("ResNet-18 fine-tuned on real brain MRI slices + brain-region cropping"
                 if _is_brain_v2 else "ResNet-18 fine-tuned on real brain MRI slices")

if os.path.exists(_brain_ckpt_path):
    print(f"[*] Loading brain MRI model ({'v2 cropped' if _is_brain_v2 else 'v1'}) ...")
    ckpt = torch.load(_brain_ckpt_path, map_location=DEVICE, weights_only=False)
    _brain_classes = ckpt["classes"]
    _brain_size = ckpt.get("size", 128)
    _brain_cropped = bool(ckpt.get("cropped", False))
    _brain = build_brain_resnet(num_classes=len(_brain_classes), pretrained=False,
                                dropout=ckpt.get("dropout", 0.0)).to(DEVICE).eval()
    _brain.load_state_dict(ckpt["state_dict"])


def predict_brain(pil_gray):
    if _brain is None:
        raise HTTPException(503, "نموذج الرنين الدماغي غير محمّل — درّب النموذج أولاً")
    im = pil_gray.convert("RGB")
    if _brain_cropped:
        # must mirror training (train_brain_v2.py) or the model sees a different distribution
        im = crop_brain_region(im)
    im = im.resize((_brain_size, _brain_size))
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = (x - _IM_MEAN_T) / _IM_STD_T
    x = x.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(_brain(x), 1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1]
    findings = []
    for i in order:
        cls = _brain_classes[i]
        p = round(float(probs[i]) * 100, 1)
        findings.append({
            "id": cls, "name_en": cls, "name_ar": BRAIN_AR.get(cls, cls),
            "probability": p, "positive": bool(i == order[0]),
            "verdict": "above" if i == order[0] else "low",
        })
    top = _brain_classes[order[0]]
    return {
        "type": "multiclass",
        "prediction_en": top, "prediction_ar": BRAIN_AR.get(top, top),
        "confidence": round(float(probs[order[0]]) * 100, 1),
        "findings": findings,
        "ood": assess_ood(probs),
        "heatmap": gradcam_overlay(_brain, x, int(order[0]), im),   # teaching heatmap
    }


REGISTRY["brain"] = {
    "meta": {
        "id": "brain", "title_ar": "الرنين المغناطيسي للدماغ — تصنيف الأورام",
        "title_en": "Brain MRI tumor classification", "modality": "brain MRI",
        "kind": "trained", "available": _brain is not None,
        "arch": "v2_resnet18_cropped" if _is_brain_v2 else "v1_resnet18",
        "source": _brain_source,
        "edu": "تصنيف أورام الدماغ في الرنين: ورم دبقي (داخل النسيج)، سحائي (من الأغشية)، نخامي (قاع الدماغ)، أو سليم. الخريطة الحرارية تساعد الطالب يربط التنبؤ بموقع الكتلة.",
        "metrics": _brain_metrics,
    },
    "predict": predict_brain,
}


# ---------------------------------------------------------------------------
# 3b) Extra MedMNIST diagnostic models (generic) — trained by train_medmnist.py.
#     Each is a ResNet-18 multiclass classifier on a real medical-imaging dataset.
# ---------------------------------------------------------------------------
MEDMNIST_MODELS = {
    "breast": {
        "title_ar": "الموجات فوق الصوتية للثدي — كشف الأورام", "title_en": "Breast ultrasound tumor",
        "modality": "breast ultrasound", "emoji": "🎗️",
        "source": "ResNet-18 on BreastMNIST (breast ultrasound; malignant vs benign)",
        "edu": "تفرّق بين الكتلة الخبيثة والحميدة في الموجات فوق الصوتية للثدي. الخبيث غالباً غير منتظم الحواف، أطول من عرضه، مع ظل خلفي؛ الحميد أملس ومحدود.",
        "class_ar": {"malignant": "خبيث", "normal, benign": "طبيعي / حميد"},
    },
    "derma": {
        "title_ar": "تحليل آفات الجلد (ديرموسكوبي)", "title_en": "Skin lesion (dermoscopy)",
        "modality": "dermoscopy", "emoji": "🩺",
        "source": "ResNet-18 on DermaMNIST (HAM10000 dermatoscopic lesions, 7 classes)",
        "edu": "٧ أنواع من آفات الجلد بالديرموسكوب. تذكّر قاعدة ABCDE للميلانوما: عدم التناظر، حدود غير منتظمة، تعدّد الألوان، القطر >6مم، التطوّر. مهمة صعبة حتى على الأطباء — استخدمها للتدريب لا للقرار.",
        "class_ar": {
            "actinic keratoses and intraepithelial carcinoma": "تقرّن سفعي / سرطان داخل ظهاري",
            "basal cell carcinoma": "سرطان الخلايا القاعدية",
            "benign keratosis-like lesions": "تقرّن حميد",
            "dermatofibroma": "ورم ليفي جلدي",
            "melanoma": "ميلانوما (خبيث)",
            "melanocytic nevi": "وحمة صبغية (شامة)",
            "vascular lesions": "آفات وعائية",
        },
    },
    "blood": {
        "title_ar": "تصنيف خلايا الدم (مجهري)", "title_en": "Blood cell classification",
        "modality": "blood smear microscopy", "emoji": "🩸",
        "source": "ResNet-18 on BloodMNIST (peripheral blood cells, 8 classes)",
        "edu": "تمييز أنواع كريات الدم في المسحة المحيطية — أساس تحليل CBC. العَدِلات تكافح البكتيريا، اللمفاويات المناعة، الحمضيات الحساسية والطفيليات. المحبّبات غير الناضجة قد تدل على عدوى شديدة أو ابيضاض.",
        "class_ar": {
            "basophil": "خلية قاعدية", "eosinophil": "خلية حمضية", "erythroblast": "أرومة حمراء",
            "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "محبّبات غير ناضجة",
            "lymphocyte": "خلية لمفاوية", "monocyte": "وحيدة", "neutrophil": "عَدِلة",
            "platelet": "صفيحة دموية",
        },
    },
    "organc": {
        "title_ar": "التعرّف على أعضاء البطن (CT)", "title_en": "Abdominal organ (CT)",
        "modality": "abdominal CT", "emoji": "🫀",
        "source": "ResNet-18 on OrganCMNIST (abdominal CT, 11 organs)",
        "edu": "تدريب على تشريح البطن في المقطعية المحوسبة (CT): تحديد العضو من شكله وموقعه وكثافته. مفيد لطالب الأشعة لبناء حسّ الموقع التشريحي قبل قراءة الحالات المرضية.",
        "class_ar": {
            "bladder": "مثانة", "femur-left": "فخذ أيسر", "femur-right": "فخذ أيمن", "heart": "قلب",
            "kidney-left": "كلية يسرى", "kidney-right": "كلية يمنى", "liver": "كبد",
            "lung-left": "رئة يسرى", "lung-right": "رئة يمنى", "pancreas": "بنكرياس", "spleen": "طحال",
        },
    },
    "path": {
        "title_ar": "أنسجة القولون (باثولوجي)", "title_en": "Colon histopathology",
        "modality": "H&E histopathology", "emoji": "🔬",
        "source": "ResNet-18 on PathMNIST (colorectal H&E tissue, 9 classes)",
        "edu": "تمييز أنسجة القولون في شرائح الباثولوجي (صبغة H&E). يشمل الظهارة السرطانية الغدية والسدى المرتبط بالورم مقابل الأنسجة الطبيعية — تدريب أساسي على قراءة الأنسجة الورمية.",
        "class_ar": {
            "adipose": "نسيج دهني", "background": "خلفية", "debris": "حطام خلوي",
            "lymphocytes": "خلايا لمفاوية", "mucus": "مخاط", "smooth muscle": "عضلات ملساء",
            "normal colon mucosa": "غشاء قولون طبيعي", "cancer-associated stroma": "سدى مرتبط بالسرطان",
            "colorectal adenocarcinoma epithelium": "ظهارة سرطان قولون غدّي",
        },
    },
    "oct": {
        "title_ar": "الشبكية — التصوير المقطعي OCT", "title_en": "Retinal OCT",
        "modality": "retinal OCT", "emoji": "👁️",
        "source": "ResNet-18 on OCTMNIST (retinal OCT, 4 classes)",
        "edu": "تصوير مقطعي للشبكية (OCT): تمييز تكوّن الأوعية المشيمية (CNV) والوذمة البقعية السكرية (DME) والدروسن عن الطبيعي — أساس متابعة أمراض الشبكية.",
        "class_ar": {
            "choroidal neovascularization": "تكوّن أوعية مشيمية (CNV)",
            "diabetic macular edema": "وذمة بقعية سكرية (DME)",
            "drusen": "دروسن (رواسب)", "normal": "طبيعي",
        },
    },
    "retina": {
        "title_ar": "اعتلال الشبكية السكري (تدريج)", "title_en": "Diabetic retinopathy grading",
        "modality": "fundus photography", "emoji": "🔆",
        "source": "ResNet-18 on RetinaMNIST (fundus, 5-grade DR)",
        "edu": "تدريج اعتلال الشبكية السكري من صور قاع العين (0=لا اعتلال ← 4=تكاثري). مهم جداً للكشف المبكر عند مرضى السكري — مهمة صعبة بدقة محدودة.",
        "class_ar": {"0": "لا اعتلال (0)", "1": "خفيف (1)", "2": "متوسط (2)", "3": "شديد (3)", "4": "تكاثري (4)"},
    },
    # --- binary screening heads -------------------------------------------------------
    # Added in the 2026-09-04 retrain pass. The 5-grade / 7-class / 4-class tasks above have
    # published accuracy ceilings well below 91%, so each also gets the *screening* question
    # a triage tool actually answers — a recognised task in its own right, not a relabelling
    # trick. The multi-class model stays served alongside. See TRAINING_LOG.md.
    "retina_bin": {
        "title_ar": "فرز اعتلال الشبكية — هل يستوجب الإحالة؟", "title_en": "Referable DR screening",
        "modality": "fundus photography", "emoji": "🚨",
        "source": "ResNet-18 on RetinaMNIST regrouped as referable DR (grade >= 2) vs not",
        "edu": "سؤال الفرز الحقيقي في برامج مسح اعتلال الشبكية السكري: هل هذي العين تحتاج إحالة لطبيب عيون؟ العتبة المعتمدة سريرياً هي الدرجة ٢ فما فوق (اعتلال متوسط أو أشد). أسهل من التدريج الخماسي ولذلك أدق — وهي المهمة الي تفيد فعلاً بالمسح الجماعي.",
        "class_ar": {"non-referable (grade 0-1)": "لا يستوجب إحالة (٠-١)",
                     "referable DR (grade 2-4)": "يستوجب الإحالة (٢-٤)"},
    },
    "derma_bin": {
        "title_ar": "فرز آفات الجلد — خبيث أم حميد؟", "title_en": "Skin lesion malignancy triage",
        "modality": "dermoscopy", "emoji": "⚠️",
        "source": "ResNet-18 on DermaMNIST regrouped as malignant/pre-malignant (akiec, bcc, mel) vs benign",
        "edu": "السؤال الي تجاوب عليه أداة فرز الديرموسكوبي: هل هذي الآفة تحتاج خزعة؟ يجمع التقرّن السفعي وسرطان الخلايا القاعدية والميلانوما بجهة واحدة مقابل الآفات الحميدة. تذكّر ABCDE. للتدريب فقط — مو بديل عن الطبيب.",
        "class_ar": {"benign (bkl/df/nv/vasc)": "حميدة",
                     "malignant or pre-malignant (akiec/bcc/mel)": "خبيثة أو ما قبل خبيثة"},
    },
    "oct_bin": {
        "title_ar": "فرز OCT — مرض شبكية أم طبيعي؟", "title_en": "OCT disease screening",
        "modality": "retinal OCT", "emoji": "🔍",
        "source": "ResNet-18 on OCTMNIST regrouped as disease (CNV/DME/drusen) vs normal",
        "edu": "قرار الإحالة في مسح OCT: هل توجد أي علامة مرضية بالشبكية (تكوّن أوعية مشيمية، وذمة بقعية، دروسن) أم المقطع طبيعي؟ خطوة الفرز الأولى قبل تحديد نوع المرض.",
        "class_ar": {"normal": "طبيعي", "disease (CNV/DME/drusen)": "مرض شبكية (CNV/DME/دروسن)"},
    },
}


def _make_medmnist_predictor(net, classes, size, class_ar, views=None, threshold=None,
                             pre=None, views_name="none"):
    """Serve a MedMNIST checkpoint in the SAME configuration its published number was measured in.

    Two mismatches lived here until 2026-09-07 (TRAINING_LOG step 54), and both made the API
    quieter than its own claims:

      TTA. The v2 recipe evaluates hflip test-time augmentation and keeps it when it helps, so
      `oct` publishes 0.9230 *with* TTA. This function did a single forward pass, i.e. 0.9180.
      Every TTA-on model was being served a fraction below its advertised accuracy.

      Threshold. tune_threshold.py picks a decision cut on val and writes it into the
      checkpoint, and the guard promotes it only when it also catches MORE disease. oct_bin has
      published 0.9920 at threshold 0.440 since session 4; argmax gives 0.9910 and catches one
      malignant case fewer. retina_bin's re-tune is larger: 0.8875 at t=0.305 catching 160/180
      referrals, against 0.8825 and 144/180 at argmax - sixteen referable eyes.

    Both are now read from the checkpoint and applied. `threshold` is only ever passed for a
    relabelled binary head, where positive is index 1 by construction; multi-class models and
    breast (whose disease class is index 0, session 3 step 14) keep argmax.
    """
    pos_idx = 1
    def predict(pil_gray):
        # Fixed preprocessing first, at the ORIGINAL resolution, because ben_graham ties its
        # blur radius to image size - running it after the resize would use a different sigma
        # than training did.
        im = preproc.apply_pil(pil_gray.convert("RGB"), pre).resize((size, size))
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - _IM_MEAN_T) / _IM_STD_T).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = preproc.tta_average(net, x, views or ["id"])[0].cpu().numpy()
        # The DECIDED class is not always the argmax once a tuned threshold is in play.
        if threshold is not None and len(classes) == 2:
            top_i = pos_idx if float(probs[pos_idx]) >= threshold else 1 - pos_idx
        else:
            top_i = int(np.argmax(probs))
        order = np.argsort(probs)[::-1]
        findings = [{
            "id": classes[i], "name_en": classes[i],
            "name_ar": class_ar.get(classes[i], classes[i]),
            "probability": round(float(probs[i]) * 100, 1),
            "positive": bool(i == top_i),
            "verdict": "above" if i == top_i else "low",
        } for i in order]
        top = classes[top_i]
        out = {
            "type": "multiclass",
            "prediction_en": top, "prediction_ar": class_ar.get(top, top),
            "confidence": round(float(probs[top_i]) * 100, 1),
            "findings": findings,
            "ood": assess_ood(probs),
            "heatmap": gradcam_overlay(net, x, int(top_i), im),   # teaching: where it looked
            "tta": views_name != "none", "tta_views": views_name,
            "preproc": pre or "none",
        }
        if threshold is not None and len(classes) == 2:
            out["decision_threshold_pct"] = round(threshold * 100, 1)
        return out
    return predict


def _make_medmnist_ensemble_predictor(members, classes, size, class_ar, threshold=None,
                                      pre=None):
    """Average several checkpoints' probabilities for one task.

    `members` is a list of (net, tta) - each member is run in ITS OWN configuration, because
    the TTA flag is a per-model decision made on that model's own val split, not a property
    of the ensemble.

    Why this exists at all. Session 2 concluded ensembles do not help, from five SEEDS of one
    resnet18 trained under GPU contention: gain over the best member was -0.0250. Session 6
    re-ran the idea with three different ARCHITECTURES on derma_bin and measured
    0.8968/0.9172/0.9117 members -> 0.9237 ensemble, with malignancies caught going
    310 -> 330. Different architectures make less correlated errors than different seeds; that
    is the whole mechanism, and it is why the earlier negative result did not generalise.

    Cost is linear: three members at 224px is three forward passes (six with TTA). Measured
    at serve time it is still well under a second on CPU for these model sizes.
    """
    pos_idx = 1
    def predict(pil_gray):
        im = preproc.apply_pil(pil_gray.convert("RGB"), pre).resize((size, size))
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - _IM_MEAN_T) / _IM_STD_T).unsqueeze(0).to(DEVICE)
        acc = None
        with torch.no_grad():
            for net, mviews in members:
                p = preproc.tta_average(net, x, mviews or ["id"])[0]
                acc = p if acc is None else acc + p
        probs = (acc / len(members)).cpu().numpy()
        if threshold is not None and len(classes) == 2:
            top_i = pos_idx if float(probs[pos_idx]) >= threshold else 1 - pos_idx
        else:
            top_i = int(np.argmax(probs))
        order = np.argsort(probs)[::-1]
        findings = [{
            "id": classes[i], "name_en": classes[i],
            "name_ar": class_ar.get(classes[i], classes[i]),
            "probability": round(float(probs[i]) * 100, 1),
            "positive": bool(i == top_i),
            "verdict": "above" if i == top_i else "low",
        } for i in order]
        top = classes[top_i]
        out = {
            "type": "multiclass",
            "prediction_en": top, "prediction_ar": class_ar.get(top, top),
            "confidence": round(float(probs[top_i]) * 100, 1),
            "findings": findings,
            "ood": assess_ood(probs),
            # Grad-CAM needs one network; the first member stands in. Saying which one it is
            # matters - a heatmap from member 1 does not explain the averaged decision, and
            # presenting it as if it did would be a nicer lie than no heatmap at all.
            "heatmap": gradcam_overlay(members[0][0], x, int(top_i), im),
            "heatmap_source": "first ensemble member only - not the averaged decision",
            "ensemble_members": len(members),
            "preproc": pre or "none",
        }
        if threshold is not None and len(classes) == 2:
            out["decision_threshold_pct"] = round(threshold * 100, 1)
        return out
    return predict


# Filled in as checkpoints load below; read by _load_quiz_pool to rebuild label groupings.
_MEDMNIST_BINARY = {}     # model_id -> original class indices counted as the positive class
_MEDMNIST_CLASSES = {}    # model_id -> class names in the model's own output order

def _load_ensemble_manifest(key):
    """models/<key>_ensemble.json, if a measured ensemble has been promoted for this task.

    A manifest is what makes an ensemble servable without editing this file per model. It is
    written by hand only after ensemble_archs.py has measured the members AND val has agreed;
    the format is deliberately dumb so it can be read and deleted by a human in a hurry:

        {"members": ["derma_bin.pt", "derma_bin_eb0.pt", "derma_bin_r50.pt"],
         "metrics": "derma_bin_ens_metrics.json",
         "why": "one sentence"}
    """
    p = os.path.join(MODEL_DIR, f"{key}_ensemble.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            man = json.load(f)
        if not man.get("members"):
            return None
        return man
    except Exception as e:
        print(f"[!] {key}: unreadable ensemble manifest ({e}) - falling back to single model")
        return None


for _key, _info in MEDMNIST_MODELS.items():
    # Prefer the v2 retrain (224px, two-stage fine-tune, TTA — see TRAINING_LOG.md) and fall
    # back to the v1 checkpoint, the same way pneumonia and brain pick their newest weights.
    _v2_path = os.path.join(MODEL_DIR, f"{_key}_v2.pt")
    _is_v2 = os.path.exists(_v2_path)
    _ckpt_path = _v2_path if _is_v2 else os.path.join(MODEL_DIR, f"{_key}.pt")
    _metrics = _load_metrics(f"{_key}_v2_metrics.json" if _is_v2 else f"{_key}_metrics.json")
    _predict = None
    _man = _load_ensemble_manifest(_key)
    if _man:
        _mem, _mck = [], None
        for _f in _man["members"]:
            _p = os.path.join(MODEL_DIR, _f)
            if not os.path.exists(_p):
                print(f"[!] {_key}: ensemble member {_f} missing - falling back to single model")
                _mem = []
                break
            _c = torch.load(_p, map_location=DEVICE, weights_only=False)
            if _mck and (_c["classes"] != _mck["classes"] or _c.get("size") != _mck.get("size")):
                print(f"[!] {_key}: member {_f} disagrees on classes or size - falling back")
                _mem = []
                break
            _n, _ = build_medmnist_backbone(_c.get("arch", "resnet18"),
                                            num_classes=len(_c["classes"]), pretrained=False,
                                            dropout=_c.get("dropout", 0.0))
            _n = _n.to(DEVICE).eval()
            _n.load_state_dict(_c["state_dict"])
            _mem.append((_n, preproc.views_for(_c)[0]))
            _mck = _mck or _c
        if _mem:
            _em = _load_metrics(_man.get("metrics", f"{_key}_ens_metrics.json")) or _metrics
            print(f"[*] Loading {_key} model (MedMNIST ENSEMBLE of {len(_mem)}: "
                  f"{', '.join(_man['members'])}) ...")
            _MEDMNIST_CLASSES[_key] = _mck["classes"]
            if _mck.get("binary_positive"):
                _MEDMNIST_BINARY[_key] = _mck["binary_positive"]
            # Same two-condition rule as the single-model path: a threshold is served only when
            # the metrics file says the published number came from it.
            _ethr = None
            if _mck.get("binary_positive") and "threshold" in str(
                    (_em or {}).get("test_accuracy_source", "")).lower():
                _ethr = _em.get("decision_threshold")
            # Members must agree on preprocessing; a mixed ensemble would feed one member a
            # distribution it never saw. Enforced above by the classes/size check plus this.
            _pres = {torch.load(os.path.join(MODEL_DIR, f), map_location="cpu",
                                weights_only=False).get("preproc", "none")
                     for f in _man["members"]}
            if len(_pres) > 1:
                print(f"[!] {_key}: members disagree on preprocessing {_pres} - falling back")
                _predict = None
            else:
                _predict = _make_medmnist_ensemble_predictor(
                    _mem, _mck["classes"], _mck.get("size", 64), _info["class_ar"],
                    threshold=_ethr, pre=_pres.pop())
            _metrics = _em
    if _predict is None and os.path.exists(_ckpt_path):
        # Checkpoints written before session 6 carry no "arch" — every one of them is a
        # ResNet-18, so that is the default. Reading it (rather than assuming) is what lets a
        # resnet50/efficientnet retrain be dropped in without touching this file again.
        _ck = torch.load(_ckpt_path, map_location=DEVICE, weights_only=False)
        _arch = _ck.get("arch", "resnet18")
        print(f"[*] Loading {_key} model (MedMNIST {_arch}{' v2' if _is_v2 else ''}) ...")
        _net, _ = build_medmnist_backbone(_arch, num_classes=len(_ck["classes"]),
                                          pretrained=False, dropout=_ck.get("dropout", 0.0))
        _net = _net.to(DEVICE).eval()
        _net.load_state_dict(_ck["state_dict"])
        _MEDMNIST_CLASSES[_key] = _ck["classes"]
        if _ck.get("binary_positive"):
            _MEDMNIST_BINARY[_key] = _ck["binary_positive"]
        # Which threshold to serve, if any. TWO conditions, and both are load-bearing:
        #
        # 1. binary_positive must be set. That marks a relabelled binary head where positive
        #    is index 1 by construction. breast_v2 is 2-class but keeps the original MedMNIST
        #    order, where disease is index 0 - applying a "positive is index 1" threshold
        #    there would tune the model to detect HEALTH, the polarity bug that cost four
        #    cancers in session 3.
        #
        # 2. The metrics file's test_accuracy_source must say the headline came from that
        #    threshold. tune_threshold.py writes a threshold into every checkpoint it touches,
        #    but the step-14 guard decides separately whether it may be published - and it
        #    REFUSES when the tuned point catches less disease. derma_bin carries
        #    threshold=0.66 for exactly that reason: it scores 0.9062 but catches 285 of 392
        #    malignancies against argmax's 316. Serving the checkpoint field blindly would
        #    have shipped the operating point the guard exists to reject.
        _thr = None
        if _ck.get("binary_positive") and "threshold" in str(
                (_metrics or {}).get("test_accuracy_source", "")).lower():
            _thr = _ck.get("threshold")
        _views, _vname = preproc.views_for(_ck)
        _predict = _make_medmnist_predictor(_net, _ck["classes"], _ck.get("size", 64),
                                            _info["class_ar"], views=_views,
                                            threshold=_thr, pre=_ck.get("preproc"),
                                            views_name=_vname)
    REGISTRY[_key] = {
        "meta": {
            "id": _key, "title_ar": _info["title_ar"], "title_en": _info["title_en"],
            "modality": _info["modality"], "kind": "trained", "available": _predict is not None,
            "emoji": _info["emoji"], "source": _info["source"], "edu": _info.get("edu"),
            "metrics": _metrics,
        },
        # predict may be None until trained; the /predict endpoint guards on `available` first
        "predict": _predict,
    }

# ---------------------------------------------------------------------------
# 4) TEXT (Arabic) — specialty router + per-category diagnosis + input filter
#    Trained on >=200k real Arabic medical entries. See ar_service.py.
# ---------------------------------------------------------------------------
import ar_service
_ar_ok = ar_service.load()
if _ar_ok:
    print("[*] Arabic symptom system loaded (router + per-category + filter).")
TEXT_META = ar_service.meta()

print("[*] All models ready:", ", ".join(list(REGISTRY.keys()) +
      (["symptoms_ar"] if _ar_ok else [])))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
# Serve the front-end page at the root so http://127.0.0.1:8000/ opens the app directly
# (same-origin as the API — avoids file:// quirks). Falls back to a JSON hint if missing.
_PAGE = os.path.join(os.path.dirname(HERE), "Radiology Hub.html")


@app.get("/")
def index():
    if os.path.exists(_PAGE):
        return FileResponse(_PAGE, media_type="text/html; charset=utf-8")
    return JSONResponse({"message": "AI Radiology Hub API",
                         "endpoints": ["/health", "/models", "/predict/{chest|pneumonia|brain}",
                                       "/predict/symptoms"],
                         "note": "Open 'Radiology Hub.html' — the page was not found next to api/."})


@app.get("/health")
def health():
    return {
        "status": "ok", "device": DEVICE,
        "models": {**{k: v["meta"].get("available", True) for k, v in REGISTRY.items()},
                   "symptoms_ar": TEXT_META["available"]},
    }


# ---- Model cards: intended use + honest limitations per model ----
INTENDED_USE = "أداة تعليمية ودعم قرار لطلاب الطب — ليست جهازاً طبياً ولا بديلاً عن الطبيب المختص أو التشخيص النهائي."
MODEL_LIMITS = {
    "chest": "مُقيّم على صور MedMNIST مصغّرة (128px)؛ الأداء على الصور كاملة الدقة قد يختلف. متوسط AUC ~0.75 = أداء متوسط، بعض الحالات (ارتشاح/نفاخ) أضعف.",
    "pneumonia": "مدرّب على أشعة أطفال (Kermany)؛ على البالغين ينخفض بوضوح (انزياح توزيع مقاس: 96٪ → 58٪). معاير لحساسية عالية → إنذارات كاذبة أكثر (نوعية أقل).",
    "brain": "التقسيم على مستوى الصورة لا المريض (مصدر واحد ~7023 صورة). قد يفشل على صور مُعزّزة/معاد ترميزها بشكل مختلف — لاحظنا انهيار صنف الورم النخامي على صور خارجية.",
    "breast": "بيانات تدريب صغيرة جداً (546 صورة) → فجوة تعميم ~9.5٪. موجات فوق صوتية فقط.",
    "derma": "مهمة صعبة (7 أصناف غير متوازنة، الشامة تسيطر)؛ دقة ~72٪ — للتدريب لا للقرار. الأصناف النادرة (ميلانوما) أصعب.",
    "blood": "دقة عالية لكن على صور مجهرية نظيفة موحّدة؛ قد يفشل مع تلطيخ/إضاءة/تكبير مختلف.",
    "organc": "تعرّف تشريحي على مقاطع CT بطن (محور واحد) — ليس كشف أمراض، بل تحديد العضو.",
    "path": "تصنيف نوع النسيج على رقع H&E صغيرة — ليس تحديد درجة/مرحلة الورم.",
    "oct": "تصنيف حالات الشبكية على OCT مصغّر؛ ليس بديلاً عن فحص قاع العين الكامل.",
    "retina": "تدريج اعتلال الشبكية السكري مهمة صعبة وبيانات قليلة (1600) → دقة محدودة؛ استخدمه لفهم صعوبة المهمة.",
    "symptoms_ar": "توجيه تخصص من نص عربي — سقف الدقة ~76٪ top-1 لتداخل التخصصات؛ يعرض أفضل 3 (top-3 ~94٪). ليس تشخيصاً.",
}


def _card(meta):
    """Enrich a model meta with model-card fields (intended use + limits) for the UI."""
    return {**meta, "intended_use": INTENDED_USE, "limits": MODEL_LIMITS.get(meta.get("id"))}


@app.get("/models")
def models():
    return {"device": DEVICE, "models": [_card(v["meta"]) for v in REGISTRY.values()] + [_card(TEXT_META)]}


class SymptomIn(BaseModel):
    text: str


@app.post("/predict/symptoms")
def predict_symptoms_ep(body: SymptomIn):
    if not ar_service.available():
        raise HTTPException(503, "نظام التنبؤ النصي العربي غير محمّل — درّبه أولاً")
    t0 = time.time()
    result = ar_service.predict(body.text)
    result.update({
        "success": True, "model_id": "symptoms",
        "model_title_ar": TEXT_META["title_ar"], "modality": TEXT_META["modality"],
        "source": TEXT_META["source"],
        "processing_time_s": round(time.time() - t0, 3),
        "disclaimer": "نتائج نموذج بحثي لدعم القرار والتعليم — لا تُستخدم للتشخيص النهائي دون طبيب مختص.",
    })
    return result


async def _read_image(file: UploadFile):
    """Decode an upload to **RGB**, and let each model reduce it if that is what it wants.

    This used to end in `.convert("L")`, which threw colour away before any model saw the
    image. Six of the fourteen models are trained on colour, and for several of them colour
    IS the signal - the H&E stain in `path`, the stain in `blood`, the pigment network in
    `derma`, haemorrhage red in `retina`. `.convert("RGB")` afterwards in the predictors only
    replicated the single grey channel three times, so the loss was total and silent.

    Measured cost of the old behaviour on the official test splits (measure_grayscale_damage.py):
        retina      0.6700 -> 0.5975   -0.0725
        retina_bin  0.8875 -> 0.8275   -0.0600, and 160/180 referrals caught fell to 126/180

    Two predictors genuinely need one channel and now say so themselves: `predict_chest`
    (torchxrayvision expects a 2-D array) and the v1 pneumonia SmallXRayCNN (in_ch=1). Every
    other model already called `.convert("RGB")` and simply gets real colour now.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "الملف فارغ")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "تعذّر قراءة الصورة — ارفع ملف صورة صالح (PNG/JPG)")


@app.post("/predict/{modality}")
async def predict(modality: str, file: UploadFile = File(...)):
    if modality not in REGISTRY:
        raise HTTPException(404, f"نموذج غير معروف: {modality}")
    entry = REGISTRY[modality]
    if not entry["meta"].get("available", True):
        raise HTTPException(503, "هذا النموذج غير متاح حالياً (لم يُدرّب بعد)")
    pil = await _read_image(file)
    t0 = time.time()
    result = entry["predict"](pil)
    result.update({
        "success": True,
        "model_id": modality,
        "model_title_ar": entry["meta"]["title_ar"],
        "modality": entry["meta"]["modality"],
        "source": entry["meta"]["source"],
        "edu": entry["meta"].get("edu"),
        "metrics": entry["meta"].get("metrics"),
        "processing_time_s": round(time.time() - t0, 2),
        "device": DEVICE,
        "filename": file.filename,
        "disclaimer": "نتائج نماذج بحثية لدعم القرار والتعليم — لا تُستخدم للتشخيص النهائي دون طبيب مختص.",
    })
    return result


# Backwards-compatible endpoint (old website used /predict for chest)
@app.post("/predict")
async def predict_legacy(file: UploadFile = File(...)):
    return await predict("chest", file)


# ---------------------------------------------------------------------------
# QUIZ MODE — teaching tool: the student sees a real labelled test image, guesses
# the diagnosis from options, then reveals the truth + the model's answer + Grad-CAM.
# ---------------------------------------------------------------------------
import base64
import random as _random

_QUIZ_POOLS = {}   # model_id -> {"imgs":[PIL], "labels":np, "classes_en":[...], "classes_ar":[...]}

# which registry models are quizzable + how to translate their classes
_PNEU_AR = ["طبيعي", "التهاب رئوي"]


def _png_b64(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_quiz_pool(model_id):
    if model_id in _QUIZ_POOLS:
        return _QUIZ_POOLS[model_id]
    import numpy as _np
    from PIL import Image as _Image
    imgs, labels, cls_en, cls_ar = [], None, [], []

    if model_id in MEDMNIST_MODELS:
        import medmnist
        from medmnist import INFO
        # A "*_bin" model is the same MedMNIST test split with its labels regrouped into the
        # screening question. The grouping lives in the checkpoint (binary_positive) so the
        # quiz can never drift from what the model was actually trained on.
        base = model_id[:-4] if model_id.endswith("_bin") else model_id
        binary_positive = _MEDMNIST_BINARY.get(model_id)
        dataset = {"organc": "organcmnist", "path": "pathmnist"}.get(base, base + "mnist")
        DataClass = getattr(medmnist, INFO[dataset]["python_class"])
        ds = DataClass(split="test", download=True, size=64, root=os.path.join(HERE, "data", "medmnist"))
        arr = ds.imgs
        labels = ds.labels.astype(int).reshape(-1)
        if binary_positive is not None:
            pos = set(binary_positive)
            labels = _np.array([1 if int(v) in pos else 0 for v in labels], dtype=int)
            cls_en = list(_MEDMNIST_CLASSES[model_id])
        else:
            cls_en = [INFO[dataset]["label"][str(i)] for i in range(len(INFO[dataset]["label"]))]
        cls_ar = [MEDMNIST_MODELS[model_id]["class_ar"].get(c, c) for c in cls_en]
        imgs = [_Image.fromarray(a) for a in arr]

    elif model_id == "pneumonia":
        import medmnist
        from medmnist import INFO
        ds = getattr(medmnist, INFO["pneumoniamnist"]["python_class"])(
            split="test", download=True, size=224, root=os.path.join(HERE, "data", "medmnist"))
        labels = ds.labels.astype(int).reshape(-1)
        cls_en = ["normal", "pneumonia"]; cls_ar = _PNEU_AR
        imgs = [_Image.fromarray(a) for a in ds.imgs]

    elif model_id == "brain" and _brain is not None:
        import pyarrow.parquet as pq
        p = os.path.join(HERE, "data", "brain_parquet", "data", "train-00000-of-00001.parquet")
        t = pq.read_table(p).to_pydict()
        labels = _np.array(t["label"], dtype=int)
        cls_en = list(_brain_classes)
        cls_ar = [BRAIN_AR.get(c, c) for c in cls_en]
        for cell in t["image"]:
            b = cell["bytes"] if isinstance(cell, dict) else cell
            imgs.append(_Image.open(io.BytesIO(b)).convert("RGB"))
    else:
        raise HTTPException(404, f"لا يتوفر اختبار لهذا النموذج: {model_id}")

    _QUIZ_POOLS[model_id] = {"imgs": imgs, "labels": labels, "classes_en": cls_en, "classes_ar": cls_ar}
    return _QUIZ_POOLS[model_id]


QUIZ_MODELS = ["brain", "pneumonia", "breast", "derma", "blood", "organc", "path", "oct", "retina",
               "derma_bin", "oct_bin", "retina_bin"]   # binary screening heads (2026-09-04 retrain)


@app.get("/quiz/models")
def quiz_models():
    out = []
    for mid in QUIZ_MODELS:
        e = REGISTRY.get(mid)
        if e and e["meta"].get("available"):
            out.append({"id": mid, "title_ar": e["meta"]["title_ar"], "emoji": e["meta"].get("emoji", "🩻")})
    return {"models": out}


@app.get("/quiz/{model}")
def quiz_case(model: str):
    if model not in QUIZ_MODELS or model not in REGISTRY or not REGISTRY[model]["meta"].get("available"):
        raise HTTPException(404, "نموذج اختبار غير متاح")
    pool = _load_quiz_pool(model)
    n = len(pool["labels"])
    idx = _random.randrange(n)
    true_idx = int(pool["labels"][idx])
    ncls = len(pool["classes_ar"])
    # 4-option MCQ (or all classes if <=4): correct + distractors, shuffled
    others = [i for i in range(ncls) if i != true_idx]
    _random.shuffle(others)
    opt_idx = [true_idx] + others[:max(1, min(3, ncls - 1))]
    _random.shuffle(opt_idx)
    options = [{"class_idx": i, "label_ar": pool["classes_ar"][i]} for i in opt_idx]
    return {"model": model, "index": idx, "image": _png_b64(pool["imgs"][idx]),
            "options": options, "n_options": len(options)}


class QuizAnswer(BaseModel):
    model: str
    index: int
    chosen_class_idx: int | None = None


@app.post("/quiz/reveal")
def quiz_reveal(body: QuizAnswer):
    if body.model not in QUIZ_MODELS or body.model not in REGISTRY:
        raise HTTPException(404, "نموذج اختبار غير متاح")
    pool = _load_quiz_pool(body.model)
    if not (0 <= body.index < len(pool["labels"])):
        raise HTTPException(400, "فهرس غير صالح")
    true_idx = int(pool["labels"][body.index])
    pil = pool["imgs"][body.index]
    t0 = time.time()
    result = REGISTRY[body.model]["predict"](pil)      # model prediction + Grad-CAM heatmap
    # model's predicted class index (map its top prediction back to the class list)
    pred_en = result.get("prediction_en", "")
    model_correct = None
    if body.model == "pneumonia":
        model_pred_idx = 1 if result.get("positive") else 0
    else:
        try:
            model_pred_idx = pool["classes_en"].index(pred_en)
        except ValueError:
            model_pred_idx = -1
    student_correct = (body.chosen_class_idx == true_idx) if body.chosen_class_idx is not None else None
    return {
        "model": body.model,
        "true_class_idx": true_idx,
        "true_label_ar": pool["classes_ar"][true_idx],
        "student_correct": student_correct,
        "model_pred_idx": model_pred_idx,
        "model_correct": (model_pred_idx == true_idx),
        "model_prediction_ar": result.get("prediction_ar") or (pool["classes_ar"][model_pred_idx] if model_pred_idx >= 0 else "—"),
        "model_confidence": result.get("confidence") or (result.get("findings", [{}])[0].get("probability")),
        "heatmap": result.get("heatmap"),
        "edu": REGISTRY[body.model]["meta"].get("edu"),
        "processing_time_s": round(time.time() - t0, 2),
    }


# ---------------------------------------------------------------------------
# CASE LIBRARY — an annotated atlas: one real labelled example per class, with the
# model's own read + Grad-CAM. Lets students build a visual reference per modality.
# ---------------------------------------------------------------------------
def _model_pred_idx(model_id, result, classes_en):
    if model_id == "pneumonia":
        return 1 if result.get("positive") else 0
    try:
        return classes_en.index(result.get("prediction_en", ""))
    except ValueError:
        return -1


@app.get("/cases/{model}")
def cases(model: str):
    if model not in QUIZ_MODELS or model not in REGISTRY or not REGISTRY[model]["meta"].get("available"):
        raise HTTPException(404, "لا تتوفر حالات لهذا النموذج")
    import numpy as _np
    pool = _load_quiz_pool(model)
    labels = pool["labels"]
    out = []
    for c in range(len(pool["classes_ar"])):
        idx_arr = _np.where(labels == c)[0]
        if len(idx_arr) == 0:
            continue
        idx = int(idx_arr[0])
        res = REGISTRY[model]["predict"](pool["imgs"][idx])
        mp = _model_pred_idx(model, res, pool["classes_en"])
        out.append({
            "class_ar": pool["classes_ar"][c],
            "image": _png_b64(pool["imgs"][idx]),
            "heatmap": res.get("heatmap"),
            "model_prediction_ar": res.get("prediction_ar"),
            "model_confidence": res.get("confidence") or (res.get("findings", [{}])[0].get("probability")),
            "model_correct": (mp == c),
        })
    return {"model": model, "title_ar": REGISTRY[model]["meta"]["title_ar"],
            "edu": REGISTRY[model]["meta"].get("edu"), "cases": out}


# ---------------------------------------------------------------------------
# REPORT-WRITING PRACTICE — student writes Findings + Impression for a hidden case,
# then compares to a structured reference report + a self-check checklist.
# ---------------------------------------------------------------------------
def _reference_report(model_id, class_ar, modality):
    return {
        "findings_ar": f"بالفحص ({modality}): مُوجودات متوافقة مع «{class_ar}».",
        "impression_ar": f"الانطباع: {class_ar}.",
        "recommendation_ar": "يُربط بالسياق السريري ورأي الأخصائي؛ متابعة أو فحوص إضافية حسب الحاجة.",
    }


def _norm_ar(s):
    import re as _re
    s = _re.sub("[ًٌٍَُِّْـ]", "", s or "")
    return (s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ة", "ه").replace("ى", "ي")).lower()


class ReportIn(BaseModel):
    model: str
    index: int
    findings: str = ""
    impression: str = ""


@app.get("/report/{model}")
def report_case(model: str):
    if model not in QUIZ_MODELS or model not in REGISTRY or not REGISTRY[model]["meta"].get("available"):
        raise HTTPException(404, "لا تتوفر حالات لهذا النموذج")
    import numpy as _np
    pool = _load_quiz_pool(model)
    idx = _random.randrange(len(pool["labels"]))
    return {"model": model, "index": idx, "image": _png_b64(pool["imgs"][idx]),
            "modality": REGISTRY[model]["meta"]["modality"], "title_ar": REGISTRY[model]["meta"]["title_ar"]}


@app.post("/report/grade")
def report_grade(body: ReportIn):
    if body.model not in QUIZ_MODELS or body.model not in REGISTRY:
        raise HTTPException(404, "نموذج غير متاح")
    pool = _load_quiz_pool(body.model)
    if not (0 <= body.index < len(pool["labels"])):
        raise HTTPException(400, "فهرس غير صالح")
    true_idx = int(pool["labels"][body.index])
    class_ar = pool["classes_ar"][true_idx]
    modality = REGISTRY[body.model]["meta"]["modality"]
    res = REGISTRY[body.model]["predict"](pool["imgs"][body.index])
    text = _norm_ar((body.findings or "") + " " + (body.impression or ""))
    key_terms = [w for w in _norm_ar(class_ar).replace("(", " ").replace(")", " ").split() if len(w) > 2]
    mentioned = any(t in text for t in key_terms) if key_terms else False
    checklist = [
        {"item_ar": "كتبتَ قسم المُوجودات (Findings)", "passed": len((body.findings or "").strip()) >= 10},
        {"item_ar": "كتبتَ الانطباع (Impression)", "passed": len((body.impression or "").strip()) >= 3},
        {"item_ar": "ذكرتَ التشخيص الصحيح", "passed": bool(mentioned)},
        {"item_ar": "التزمتَ ببنية التقرير (موجودات + انطباع)",
         "passed": len((body.findings or "").strip()) >= 10 and len((body.impression or "").strip()) >= 3},
    ]
    return {
        "model": body.model, "true_label_ar": class_ar,
        "reference": _reference_report(body.model, class_ar, modality),
        "model_prediction_ar": res.get("prediction_ar"),
        "model_confidence": res.get("confidence") or (res.get("findings", [{}])[0].get("probability")),
        "heatmap": res.get("heatmap"),
        "edu": REGISTRY[body.model]["meta"].get("edu"),
        "checklist": checklist,
        "score": sum(1 for c in checklist if c["passed"]),
        "max_score": len(checklist),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
