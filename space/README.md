---
title: AI Radiology Hub API
emoji: 🩻
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: FastAPI serving 14 trained medical-imaging and Arabic NLP models
---

# AI Radiology Hub — serving API

FastAPI serving 14 trained models for the front-end at
<https://ai-powered-radiology-hub.vercel.app>. Nothing here is simulated: every model runs
on real medical data and every number `/models` reports was measured on a held-out test
split. This Space exists because the page is static and the models are 1.4 GB — the page
hosts anywhere, the models need a machine.

**Educational and decision-support only. Not a medical device, and not a substitute for a
clinician.** Every model card returned by `/models` carries its own measured limitations,
including the ones that are unflattering.

## Endpoints

| | |
|---|---|
| `GET /health` | which models loaded, and the device |
| `GET /models` | model cards: measured metrics, intended use, limitations |
| `POST /predict/{model}` | image upload → prediction + Grad-CAM |
| `POST /predict/symptoms` | Arabic symptom text → specialty routing |
| `GET /quiz/{model}`, `GET /cases/{model}`, `POST /report/grade` | teaching modes |

## Honest numbers

Accuracy on held-out test data, generated from the server itself:

| Model | Accuracy | Model | Accuracy |
|---|---|---|---|
| `oct_bin` | 0.992 | `derma_bin` | 0.934 (3-model ensemble) |
| `brain` | 0.990 | `oct` | 0.923 |
| `blood` | 0.988 | `retina_bin` | 0.888 |
| `pneumonia` | 0.963 | `derma` | 0.879 |
| `path` | 0.955 | `breast` | 0.878 |
| `organc` | 0.953 | `retina` | 0.670 |

`chest` is multi-label and scored by AUC (0.7525 mean), not accuracy. `retina` at 0.670 is
published as measured: RetinaMNIST's best published result is ~0.53, and the four hardest
tasks here have scientific ceilings well below 90%. Those rows are not failures being
hidden — they are what the task actually allows.

## Limits worth knowing

- **CPU only.** Inference is ~0.1–0.8 s per image.
- **The Space sleeps** after inactivity on the free tier; the first request after a sleep
  pays a cold start.
- `chest` downloads its weights from torchxrayvision on first use, so the first chest
  request after a cold start is slower than the rest.

Full training history, including the measured negative results and the bugs that were
caught: `TRAINING_LOG.md` in the project repository.
