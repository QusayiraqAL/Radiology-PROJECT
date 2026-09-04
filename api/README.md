# AI Radiology Hub — Real Models Backend

Four **real, trained-and-validated** systems served over FastAPI. Nothing here is
simulated: every model runs on real medical data and every performance number is
measured on a held-out test split.

> **We publish the generalization gap (train accuracy − test accuracy) for every model
> we train.** A model that scores well on test but perfectly on train is memorizing, and
> hiding that helps nobody. The gap is in `*_metrics.json`, in `/models`, and on the site.

> 📓 **This file describes where the project *is*. [`../TRAINING_LOG.md`](../TRAINING_LOG.md)
> describes how it *got here*** — a dated, run-by-run record of every training pass: why it
> was run, what changed, and the measured before/after. Read that one to understand a
> decision; read this one to look up a model.

## ⚠️ Data-leakage finding (brain MRI) — read this first

The brain model's original **98.95%** was **not a real generalization estimate**. Measured
with `check_brain_leakage.py` (perceptual dHash, Hamming distance to nearest train image):

| Check on the old per-image test split | Result |
|---|---|
| Test images with an **exact** hash twin in train | **22.0%** |
| Test images with a near-twin (Hamming ≤ 3) | **43.2%** |
| Test images with a near-twin (Hamming ≤ 5) | **66.3%** |

Cause: consecutive MRI slices of one patient are near-identical, this public dataset ships
**no patient IDs**, and the split was per-**image** — so the same patient landed on both
sides of the boundary.

Fix: `brain_split.py` groups near-duplicate slices into clusters (union-find over dHash,
Hamming ≤ 5) and splits by **cluster**, so a patient/acquisition can never straddle the
split. Verified with `_verify_split.py`: **0.0% duplicates at every threshold** (mean min
distance 4.18 → 8.29).

### The result refuted the hypothesis — and that matters

We expected the honest number to be **lower**. It wasn't:

| | v1 (contaminated per-image split) | **v2 (leak-free grouped split)** |
|---|---|---|
| Test accuracy | 98.95% (n=1052) | **99.00%** (n=699) |
| Macro-AUC | 0.9994 | 0.9993 |
| Train/test gap | not measured | **+0.009** |
| Mean confidence | 100.0% on samples | **95.5%** (label smoothing) |

So the contamination was **real** (22% exact twins is a measured fact), but it was **not what
drove the score** — the model genuinely separates these four classes (7 errors in 699). The
original 98.95% turned out to be trustworthy.

**The point stands anyway:** we only know it was trustworthy *because we checked*. An
uninvestigated 98.95% on a split with 66% near-duplicates is not evidence of a good model —
it's an unfalsified guess that happened to be right. The check is what converts it into a
result you can defend.

## ⚠️ Second finding: the input filter's 99.8% was a shortcut

The Arabic input filter reported **99.76% accuracy / 99.79% non-medical rejection**. That
number was real *on its own test split* — and still misleading. Its training negatives were
Arabic **reviews + news** (all declarative prose) while every positive was a **question**, so
it could score ~99% by learning *"is this a question?"* instead of *"is this medical?"*.

Measured by `check_filter_robustness.py` on question-form non-medical Arabic:

| Filter | Claimed (own test set) | **Question-form non-medical (real)** | Medical recall |
|---|---|---|---|
| v1 | 99.79% | **33.3%** — 8/12 leaked through | 100% |
| **v2** (`ar_filter_v2.py`) | 99.7% | **98.2%** held-out · **83.3%** independent probes | **100%** |

Fix: add **real** question-form non-medical Arabic as hard negatives — `hsseinmz/arcd` and
`google/xquad` (`xquad.ar`), i.e. Wikipedia questions. No synthetic text. v2 publishes
`holdout_question_negative_rejection_rate` (rejection on questions it never trained on)
**alongside** the in-distribution score, so the metric can't flatter itself again.

> Both findings share one lesson: **a high score against an easy or contaminated
> evaluation set is not a high score.** Check what the test set actually contains.

## Models

| id | Modality | Task | Backbone | Data | Validation |
|----|----------|------|----------|------|------------|
| `chest` | Chest X-ray | 18-pathology multi-label | DenseNet121 (TorchXRayVision, pretrained on NIH/CheXpert/MIMIC/PadChest/…) | — | External: per-label ROC-AUC on **ChestMNIST** test (NIH ChestX-ray14, 224px) |
| `pneumonia` | Chest X-ray | Normal vs Pneumonia | **v2:** ResNet-18 @224 (ImageNet transfer, dropout+label smoothing) | **PneumoniaMNIST-224** (Kermany 2018 pediatric CXR, real full-res) | Accuracy / AUC / sensitivity / specificity + **gap** on official test split |
| `brain` | Brain MRI | glioma / meningioma / pituitary / no-tumor | **v2:** ResNet-18 + brain-region crop (ImageNet init) | **Brain Tumor MRI Dataset** (HuggingFace, real MRI slices) | Accuracy / macro-F1 / macro-AUC + **gap**, on a **leak-free grouped** test split (see finding above) |
| `symptoms_ar` | Clinical text (Arabic) | Arabic symptom text → specialty (20) → diagnosis | **Router** (SGD, L2 swept) + **per-category** (calibrated LinearSVC) + **input filter v2**; TF-IDF word+char | **≥200k real Arabic** (Shifaa + hajerbchn/MAQA + MKamil) | Router top-1 / top-3 + **gap**, per-category acc, filter reject-rate on **unseen question-form** negatives |
| `breast` | Breast ultrasound | malignant vs benign | ResNet-18 (transfer) | **BreastMNIST** (real breast US) | test acc **0.808**, AUC **0.868**, + gap |
| `derma` | Dermoscopy | 7-class skin lesion (incl. melanoma) | ResNet-18 (transfer) | **DermaMNIST** (HAM10000) | test acc **0.722**, macro-AUC **0.925**, + gap (hard imbalanced task) |
| `blood` | Blood-smear microscopy | 8-class blood cell | ResNet-18 (transfer) | **BloodMNIST** (real peripheral blood) | test acc **0.979**, macro-AUC **0.999**, gap **0.006** |
| `organc` | Abdominal CT | 11-class organ identification | ResNet-18 (transfer) | **OrganCMNIST** (abdominal CT) | test acc **0.942**, macro-AUC **0.993** |
| `path` | H&E histopathology | 9-class colorectal tissue | ResNet-18 (transfer) | **PathMNIST** (colorectal H&E) | test acc **0.944**, macro-AUC **0.996** |
| `oct` | Retinal OCT | 4-class (CNV/DME/drusen/normal) | ResNet-18 (transfer) | **OCTMNIST** | test acc **0.775**, macro-AUC **0.967** |
| `retina` | Fundus | 5-grade diabetic retinopathy | ResNet-18 (transfer) | **RetinaMNIST** | test acc **0.495** (hard task, ~benchmark), AUC 0.772 |

### Safety & transparency features
- **OOD / low-confidence flag** (`assess_ood`): every image prediction carries an `ood` block
  (max-softmax + normalized entropy) that warns when the image may be the wrong modality or an
  unusual case. It is a **heuristic, not a guarantee** — nets can be over-confident on garbage
  (a good teaching point); the model card says so.
- **Model cards** (`/models` returns `intended_use` + `limits` per model): honest, per-model
  "what it's for and where it fails" — shown as an expandable "🪪 بطاقة النموذج" in the UI.
- **Grad-CAM++** (`gradcam.py`): upgraded from vanilla Grad-CAM for sharper localization.

### Teaching features (for training medical students)
- **Grad-CAM heatmaps** (`gradcam.py`): every ResNet image model returns a `heatmap` (base64 PNG)
  showing *where* the network looked — the core teaching aid ("does the hot region match the real
  finding?"). Served in `/predict` responses, shown in the UI.
- **Arabic educational notes** (`edu` field per model): a short "what is this / how to read it"
  blurb (ABCDE for melanoma, ABCDE for CXR, CBC cell roles, etc.), shown in results.
- **Quiz mode** (`/quiz/{model}`, `/quiz/reveal`): serves a real *labelled* test image with a
  4-option MCQ; the student guesses, then the reveal shows the correct answer, the model's own
  prediction (right or wrong), the Grad-CAM heatmap, and the educational note — with a running
  "you vs the model" scoreboard. The answer is never sent to the client before answering.
- **Case library / atlas** (`/cases/{model}`): one real labelled example per class with the
  original + Grad-CAM side by side and the model's read — a visual reference per modality.
- **Report-writing practice** (`/report/{model}`, `/report/grade`): the student writes
  Findings + Impression for a hidden case, then gets a structured reference report, the true
  diagnosis, the model's read + heatmap, and a checklist score on report structure.
  All three teaching tools cover: brain, pneumonia, breast, derma, blood, organ-CT, path, OCT, DR.

The three `breast`/`derma`/`blood` models are trained by the generic `train_medmnist.py`
(same anti-overfitting recipe: transfer learning, augmentation, dropout, label smoothing,
weight decay, early stop on val loss, class weights) and auto-discovered by `main.py`
(`MEDMNIST_MODELS`). Retrain any of them with e.g. `DATASET=bloodmnist python train_medmnist.py`.

### Arabic text system details (`ar_*.py`)
- **≥200,000 genuine Arabic entries** aggregated (232k corpus; 196k category-labelled).
- `ar_data_prep.py` → download/clean/dedupe/normalize (see `data/arabic/stats.json`).
- `ar_train_v2.py` → specialty router (20) + per-category models. **Use this**; `ar_train.py` is the v1 baseline kept for comparison.
- `ar_filter_v2.py` → input filter trained with **question-form hard negatives**. **Use this**; `ar_filter.py` is v1 (see the shortcut finding above).
- `ar_service.py` → serving pipeline: rule gate → filter → router → per-category diagnosis, with **abstain** on low confidence and **specialist referral** when no fine model exists.
- `ar_train_transformer.py` → **Colab/GPU** MARBERT fine-tuning (plug-in upgrade for the router).

**Measured (held-out, v1 → v2 → MARBERT):**

| | v1 (linear) | v2 (linear) | **MARBERT (active)** |
|---|---|---|---|
| Router top-1 | 0.7048 | 0.7053 | **0.756** |
| Router **top-3** | 0.8923 | 0.9011 | **0.936** |
| Router train/test gap | +0.215 | +0.141 | **+0.128** |
| Per-category accuracy | 0.616–0.909 | 0.612–0.910 (mean **+0.33pt**) | (unchanged — per-category still linear) |
| Filter — unseen question-form negatives | 0.333 | **0.982** | 0.982 |

**MARBERT is now the served router.** `ar_service.py` auto-detects `models/ar/marbert_router/`
(UBC-NLP/MARBERT fine-tuned on Colab GPU via `MARBERT_Colab_Train.ipynb`) and routes with it,
falling back to the linear model if the folder or `transformers` is absent. `/models` reports
`router_backend: marbert | linear`. MARBERT lifted top-1 **0.705 → 0.756** and top-3
**0.901 → 0.936** on the same held-out split — real gains, though still short of 95% because
the 20 specialties genuinely overlap (see below).

**Why the router tops out near 70% top-1** — and why top-3 is the number that matters:
the 20 specialties genuinely overlap in patient-written Q&A (`general_medicine` vs
`internal_medicine`, `dermatology` vs `cosmetic_surgery`), and the labels are user-chosen,
not adjudicated. Every bag-of-words classifier we swept lands in the same 0.65–0.71 band
(LinearSVC, LogReg/saga, ComplementNB, SGD) — that's a **data ceiling, not a tuning bug**.
So the router **abstains** below 30% confidence and surfaces the **top-3**, which is right
~90% of the time. Breaking the ceiling needs a transformer (MARBERT):
`ar_train_transformer.py` is ready for a GPU box — **this machine is CPU-only**, where
fine-tuning 166k examples is not viable.

### Pneumonia v1 → v2 (measured on the official 624-image test split)

| | v1 (SmallCNN @64) | **v2 (ResNet-18 @224)** |
|---|---|---|
| Accuracy | 0.8974 | **0.9631** |
| AUC | 0.9584 | **0.9941** |
| Sensitivity | 0.9923 | 0.9923 |
| **Specificity** | **0.7393** | **0.9145** |
| Train/test gap | not measured | **+0.031** |

v1's real weakness was **specificity 0.739** — it caught pneumonia but called too many
healthy chests sick (61 false positives of 234 normals). v2 cuts that to 20 while holding
sensitivity identical, and its threshold is picked by Youden's J on validation, not guessed.

## Anti-overfitting (what's actually applied)

| Technique | pneumonia v2 | brain v2 | Arabic router |
|---|---|---|---|
| Transfer learning (ImageNet ResNet-18) | ✅ | ✅ | — |
| Data augmentation | ✅ crop/zoom/shear/flip/brightness | ✅ rotate/flip/translate/scale/brightness | — |
| Dropout head | ✅ 0.3 | ✅ 0.3 | — |
| Label smoothing | ✅ 0.05 | ✅ 0.05 | — |
| Weight decay / L2 | ✅ 1e-4 | ✅ 1e-4 | ✅ alpha **swept**, not guessed |
| Early stopping + best checkpoint | ✅ on val **loss** | ✅ on val **loss** | — |
| Class weights (balanced) | ✅ | ✅ | ✅ |
| Leak-free grouped split | — | ✅ **the big one** | — |
| Soft-vote ensemble | — | — | ✅ router only |
| **Train/test gap published** | ✅ | ✅ | ✅ |

Two notes on decisions that were **measured, not assumed**:
- **Router L2**: sweeping alpha 2e-5 → 5e-5 cut the train/test gap **0.215 → 0.141** while
  top-3 *rose* 0.893 → 0.904 and top-1 moved only −0.1pt. Stronger still (2e-4/5e-4) traded
  away too much accuracy.
- **Per-category**: the soft-vote SGD that helps the router **hurts** here — those sets are
  small (260–10k rows vs 120k features), so SGD hits train=1.000 and scores *below* v1.
  Per-category therefore keeps a calibrated LinearSVC. Same idea, opposite conclusion,
  because the data is different. Selecting on val **loss** (not val accuracy) also matters:
  accuracy plateaus and silently keeps the most over-confident epoch.

## Files

- `main.py` — FastAPI server (model registry + endpoints); prefers **v2** checkpoints, falls back to v1
- `nets.py` — shared model architectures (training **and** serving — one source of truth)
- `ar_models.py` — shared Arabic text classes (`TfidfUnion`, `SoftVoteText`); joblib pickles by
  module reference, so these must live outside the training script or the API can't load them
- `brain_split.py` — dHash + near-duplicate clustering + **grouped** (leak-free) split
- `check_brain_leakage.py` — measures train/test contamination; writes `brain_leakage_report.json`
- `_verify_split.py` — proves the grouped split removes it; writes `split_comparison.json`
- `img_utils.py` — `crop_brain_region` (applied identically at train **and** serve time)
- `train_pneumonia_v2.py` / `train_brain_v2.py` — v2 training with the table above
- `ar_train_v2.py` — Arabic router (ensemble) + per-category models, with gap reporting
- `validate_chest.py` — external validation of the pretrained chest model
- `validate_all.py` — consolidated report (prefers v2 metrics, prints the gap)
- `models/*.pt` — trained weights · `models/*_metrics.json` — measured metrics (served by `/models`)

## GPU (optional) — breaking the Arabic router's 70% ceiling

Everything above was trained **CPU-only**. The one thing a GPU unlocks that CPU can't is
fine-tuning **MARBERT** for the router — the only realistic way past the 0.65–0.71 bag-of-words
ceiling. Notes for this machine (GTX 1650, 4 GB, compute 7.5):

- **Driver 511.65 (2022) caps you at CUDA 11.x.** CUDA 12 wheels need ≥ 527.41, so use the
  **cu118** line — which tops out at **torch 2.7.1** (a downgrade from 2.13). Updating the
  NVIDIA driver removes this constraint and lets you use current torch.
- **Disk**: the cu118 wheel is 2.82 GB and extracts to **5.45 GB** → **~8.27 GB peak** during
  `pip install`. Measure free space first; this drive was 99% full.

```bash
# with driver >= 452.39 (CUDA 11.8 line)
pip uninstall -y torch torchvision
pip install --no-cache-dir torch==2.7.1+cu118 torchvision==0.22.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

pip install transformers datasets accelerate
python ar_train_transformer.py      # defaults tuned for 4 GB: batch 16 x accum 2, fp16, grad-checkpointing
# -> writes models/ar/marbert_router/ ; ar_service.py picks it up automatically
```

`ar_service.py` genuinely auto-detects `models/ar/marbert_router/` and routes with it, falling
back to the linear router if it is missing or unloadable. `/models` reports `router_backend`
(`marbert` | `linear`) so you can confirm which one actually answered.

**Expectation-setting:** the 1650 has **no tensor cores**, so fp16 mainly saves VRAM rather
than time — fine-tuning 166k examples is an hours-long job. On a Colab T4 it's much faster
(raise `BATCH=32`).

## Endpoints

- `GET  /health` — server + per-model availability
- `GET  /models` — metadata + **real measured metrics** for every model, incl. `router_backend`
- `POST /predict/{chest|pneumonia|brain}` — upload an image, get real predictions
- `POST /predict/symptoms` — JSON `{"text": "..."}`, returns category → disease + confidence
- `POST /predict` — legacy alias for `/predict/chest`

## Reproduce from scratch

```bash
# from the api/ folder, with the venv python
python data_prep.py            # download MedMNIST (chest datasets)

# --- images (v2 = transfer learning + augmentation + dropout + label smoothing + early stop)
python train_pneumonia_v2.py   # ResNet-18 @224 on real Kermany CXR
python check_brain_leakage.py  # FIRST: measure train/test contamination (writes a report)
python _verify_split.py        # prove the grouped split removes it
python train_brain_v2.py       # ResNet-18 + crop, on the LEAK-FREE grouped split
python validate_chest.py       # externally validate the pretrained chest model

# --- Arabic text
python ar_data_prep.py         # aggregate >=200k real Arabic medical data
python ar_train_v2.py          # router (ensemble) + per-category models  [ROUTER_ALPHA, CAT_C]
python ar_filter.py            # input filter (flags unsatisfactory inputs)

python validate_all.py         # consolidated validation_report.json (prefers v2, prints gaps)
uvicorn main:app --host 127.0.0.1 --port 8000
```

`main.py` auto-detects `pneumonia_v2.pt` / `brain_tumor_mri_v2.pt` and serves them, falling
back to the v1 checkpoints if they're absent — so the server keeps working mid-retrain.
`ar_train_v2.py` accepts `AR_SAMPLE=2000` for a fast smoke test (it writes real artifacts,
so back up `models/ar/` first).

Or just double-click `start_server.bat` in the project root (it runs the server;
the training scripts only need to be run once to produce `models/`).

## Disclaimer

Research/education decision-support only. Not a medical device; not for clinical
diagnosis. A licensed radiologist is always the final authority.
