# -*- coding: utf-8 -*-
"""
Hybrid experiment: classical ML heads on top of a trained CNN, vs the CNN's own softmax.

THE QUESTION. "Why not combine classical algorithms with deep learning?" There is a real
answer and a fashionable non-answer, and only measurement tells them apart. So this script
measures three feature sets against four families of classical head, plus the CNN itself:

  cnn      512-d penultimate ResNet-18 features (the layer that feeds `fc`)
  texture  ~110 classical descriptors computed on the raw pixels: GLCM/Haralick at 3
           distances x 4 angles, two LBP histograms, and intensity statistics. For breast
           ultrasound these are not arbitrary - margin irregularity and internal echo
           texture are what a radiologist actually reads, and a 224px CNN can miss them.
           This is the half that carries information the CNN does not already have.
  both     concatenation of the two

The honest prior: swapping `fc` for an SVM usually gains ~nothing, because `fc` is already a
linear classifier on those same 512 features. The case where it can genuinely help is small
data - BreastMNIST has 546 training images - where a strongly regularized classical head
generalizes better than a fully fine-tuned linear layer. `texture` is the part that could
add something new. All three are reported so the difference is visible rather than assumed.

PROTOCOL (this is what makes the numbers publishable):
  - every model and every hyperparameter is chosen on the VALIDATION split;
  - the decision threshold for binary tasks is also chosen on validation;
  - the TEST split is touched exactly once, at the end, to report the selected model.
  No test-set peeking anywhere, which is the only reason a gain here would mean anything.

Runs on CPU on purpose: the GPU is busy training, and this costs minutes at this data size.

  python hybrid_heads.py              # breast_v2
  KEY=derma_bin python hybrid_heads.py
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

from nets import build_brain_resnet

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
KEY = os.environ.get("KEY", "breast_v2")
DEVICE = os.environ.get("HYBRID_DEVICE", "cpu")   # cpu by default: the GPU is training
SEED = 0
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Checkpoints saved before the `medmnist` field was added carry no dataset name, which is
# exactly what makes tune_threshold.py skip breast_v2. Backfilling it here is that fix.
KEY_TO_DATASET = {"breast_v2": "breastmnist", "retina_v2": "retinamnist",
                  "derma_v2": "dermamnist", "oct_v2": "octmnist",
                  "retina_bin": "retinamnist", "derma_bin": "dermamnist",
                  "oct_bin": "octmnist", "breast": "breastmnist"}
BINARY_POSITIVE = {"retina_bin": [2, 3, 4], "derma_bin": [0, 1, 4], "oct_bin": [0, 1, 2]}

# WHICH CLASS INDEX IS THE DISEASE. Do not assume 1. The `_bin` heads are built by remapping
# original labels onto {0,1} with disease=1, so 1 is right for them. BreastMNIST is NOT: its
# native order is ['malignant', 'normal, benign'], so the disease sits at index 0 and
# `p[:, 1]` is P(benign). Getting this backwards does not raise - it silently optimises the
# threshold for detecting HEALTH, which on breast_v2 measurably cost 4 extra missed cancers
# while the accuracy number went UP. Every sensitivity below is w.r.t. this index.
CLINICAL_POSITIVE = {"breast_v2": 0, "breast": 0}     # default 1 for everything else
MIN_SENS = float(os.environ.get("MIN_SENS", "0.90"))


def log(m):
    print(m, flush=True)


class DS(Dataset):
    def __init__(self, X, size):
        self.X = X
        self.tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        im = self.X[i]
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        return self.tf(np.ascontiguousarray(im))


def load_ckpt():
    """Load the checkpoint, backfilling the dataset fields older runs did not save."""
    path = os.path.join(MODEL_DIR, KEY + ".pt")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    changed = False
    if not ck.get("medmnist"):
        ck["medmnist"] = KEY_TO_DATASET[KEY]
        changed = True
    if "binary_positive" not in ck:
        ck["binary_positive"] = BINARY_POSITIVE.get(KEY)
        changed = True
    if changed:
        torch.save(ck, path)
        log("[fix] backfilled medmnist=%s binary_positive=%s into %s.pt"
            % (ck["medmnist"], ck["binary_positive"], KEY))
    return ck


def load_split(dataset, split, size, positive):
    DataClass = getattr(medmnist, INFO[dataset]["python_class"])
    ds = DataClass(split=split, download=True, size=size, root=DATA_ROOT)
    y = ds.labels.astype(np.int64).reshape(-1)
    if positive:
        pos = set(positive)
        y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)
    return ds.imgs, y


@torch.no_grad()
def cnn_features(net, fc, X, size, tta):
    """One pass gives both the 512-d penultimate features AND the CNN's own logits: in eval
    mode the head is a deterministic function of those features (dropout is the identity)."""
    feats, logits = [], []
    for xb in DataLoader(DS(X, size), batch_size=32):
        xb = xb.to(DEVICE)
        f = net(xb)
        if tta:
            f = (f + net(torch.flip(xb, dims=[3]))) / 2
        feats.append(f.cpu().numpy())
        logits.append(fc(f).cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def texture_features(X, tag):
    """GLCM/Haralick + LBP + intensity statistics on the raw pixels - the classical half."""
    dists, angles = [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    out = []
    for k, im in enumerate(X):
        g = im if im.ndim == 2 else (im[..., :3] @ np.array([0.299, 0.587, 0.114]))
        g = np.ascontiguousarray(g).astype(np.uint8)
        q = (g // 8).astype(np.uint8)                      # 32 grey levels: stabler GLCM
        m = graycomatrix(q, distances=dists, angles=angles, levels=32,
                         symmetric=True, normed=True)
        v = [graycoprops(m, p).ravel() for p in props]     # 6 props x 3 dist x 4 ang = 72
        for P, R in ((8, 1), (16, 2)):                     # 10 + 18 = 28 LBP bins
            lbp = local_binary_pattern(g, P, R, method="uniform")
            h, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2), density=True)
            v.append(h)
        gf = g.astype(np.float32)
        sd = gf.std() + 1e-9
        v.append(np.array([gf.mean(), gf.std(),
                           *np.percentile(gf, [5, 25, 50, 75, 95]),
                           float(((gf - gf.mean()) ** 3).mean() / sd ** 3),
                           float(((gf - gf.mean()) ** 4).mean() / sd ** 4)]))
        out.append(np.concatenate(v))
        if (k + 1) % 1000 == 0:
            log("      texture %s %d/%d" % (tag, k + 1, len(X)))
    return np.nan_to_num(np.stack(out).astype(np.float32))


def heads():
    """Small grids only. Every one of these is selected on validation, never on test."""
    h = []
    for C in (0.01, 0.1, 1.0, 10.0):
        h.append(("logreg C=%g" % C,
                  make_pipeline(StandardScaler(),
                                LogisticRegression(C=C, max_iter=3000,
                                                   class_weight="balanced"))))
    for C in (0.1, 1.0, 10.0):
        h.append(("svm-rbf C=%g" % C,
                  make_pipeline(StandardScaler(),
                                SVC(C=C, kernel="rbf", gamma="scale", probability=True,
                                    class_weight="balanced", random_state=SEED))))
    for d in (None, 8):
        h.append(("rf depth=%s" % d,
                  RandomForestClassifier(n_estimators=500, max_depth=d, n_jobs=-1,
                                         class_weight="balanced", random_state=SEED)))
    for lr in (0.05, 0.1):
        h.append(("hgb lr=%g" % lr,
                  HistGradientBoostingClassifier(learning_rate=lr, max_iter=300,
                                                 early_stopping=True, random_state=SEED)))
    return h


def binary_rates(y, pred, pos):
    """Rates with respect to the DISEASE class `pos`, not blindly class 1."""
    yb, pb = (y == pos).astype(int), (pred == pos).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, pb, labels=[0, 1]).ravel()
    return {"positive_class_index": int(pos),
            "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
            "specificity": round(float(tn / max(tn + fp, 1)), 4),
            "disease_caught": "%d/%d" % (int(tp), int(tp + fn)),
            "confusion_matrix_[[tn,fp],[fn,tp]]": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    t0 = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    ck = load_ckpt()
    dataset, size = ck["medmnist"], ck.get("size", 224)
    classes, tta = ck["classes"], bool(ck.get("tta", False))
    n_cls = len(classes)
    log("[hybrid] key=%s dataset=%s size=%d classes=%d device=%s tta=%s"
        % (KEY, dataset, size, n_cls, DEVICE, tta))

    Xtr, ytr = load_split(dataset, "train", size, ck.get("binary_positive"))
    Xva, yva = load_split(dataset, "val", size, ck.get("binary_positive"))
    Xte, yte = load_split(dataset, "test", size, ck.get("binary_positive"))
    log("[hybrid] train=%d val=%d test=%d" % (len(ytr), len(yva), len(yte)))

    net = build_brain_resnet(num_classes=n_cls, pretrained=False,
                             dropout=ck.get("dropout", 0.0))
    net.load_state_dict(ck["state_dict"])
    fc = net.fc
    net.fc = nn.Identity()
    net = net.to(DEVICE).eval(); fc = fc.to(DEVICE).eval()

    log("[hybrid] extracting CNN features ...")
    Ftr, Ltr = cnn_features(net, fc, Xtr, size, tta)
    Fva, Lva = cnn_features(net, fc, Xva, size, tta)
    Fte, Lte = cnn_features(net, fc, Xte, size, tta)
    log("[hybrid] cnn feature dim = %d" % Ftr.shape[1])

    log("[hybrid] computing classical texture features ...")
    Ttr = texture_features(Xtr, "train")
    Tva = texture_features(Xva, "val")
    Tte = texture_features(Xte, "test")
    log("[hybrid] texture feature dim = %d" % Ttr.shape[1])

    # ---- CNN baseline: the model exactly as it is served --------------------------------
    Pva_cnn, Pte_cnn = softmax(Lva), softmax(Lte)
    base_val = accuracy_score(yva, Pva_cnn.argmax(1))
    base_test = accuracy_score(yte, Pte_cnn.argmax(1))
    log("\n[baseline] CNN softmax  val=%.4f  test=%.4f" % (base_val, base_test))

    sets = {"cnn": (Ftr, Fva, Fte),
            "texture": (Ttr, Tva, Tte),
            "both": (np.hstack([Ftr, Ttr]), np.hstack([Fva, Tva]), np.hstack([Fte, Tte]))}

    results, best = [], None
    for sname, (A, B, _C) in sets.items():
        log("\n[set] %s (dim=%d)" % (sname, A.shape[1]))
        for hname, clf in heads():
            try:
                clf.fit(A, ytr)
                acc_val = accuracy_score(yva, clf.predict_proba(B).argmax(1))
            except Exception as e:
                log("   %-16s FAILED %s: %s" % (hname, type(e).__name__, e)); continue
            log("   %-16s val=%.4f" % (hname, acc_val))
            rec = {"features": sname, "head": hname, "val_accuracy": round(float(acc_val), 4)}
            results.append(rec)
            if best is None or acc_val > best["val_accuracy"]:
                best = dict(rec); best_clf = clf

    log("\n[select] best on VALIDATION: %s + %s  val=%.4f"
        % (best["features"], best["head"], best["val_accuracy"]))

    # ---- the single test measurement ----------------------------------------------------
    _A, Bsel, Csel = sets[best["features"]]
    pv, pt = best_clf.predict_proba(Bsel), best_clf.predict_proba(Csel)
    test_acc = accuracy_score(yte, pt.argmax(1))
    out = {
        "model": "%s_hybrid" % KEY, "dataset_key": KEY, "medmnist": dataset,
        "question": "do classical heads on CNN features beat the CNN's own softmax?",
        "protocol": ("model, hyperparameters and threshold all selected on the validation "
                     "split; the test split was measured once, at the end"),
        "cnn_baseline": {"val_accuracy": round(float(base_val), 4),
                         "test_accuracy": round(float(base_test), 4)},
        "selected": {"features": best["features"], "head": best["head"],
                     "val_accuracy": best["val_accuracy"],
                     "test_accuracy": round(float(test_acc), 4)},
        "delta_vs_cnn": round(float(test_acc - base_test), 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(yte, pt.argmax(1))), 4),
        "test_macro_f1": round(float(f1_score(yte, pt.argmax(1), average="macro")), 4),
        "all_val_results": sorted(results, key=lambda r: -r["val_accuracy"]),
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "test_split": "full official MedMNIST test split (never subsampled, never tuned on)",
        "device": DEVICE, "seed": SEED,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if n_cls == 2:
        pos = CLINICAL_POSITIVE.get(KEY, 1)
        neg = 1 - pos
        log("\n[polarity] disease class = index %d (%s); scoring P(class %d)"
            % (pos, classes[pos], pos))
        sv, st = pv[:, pos], pt[:, pos]            # P(disease) on val and on test
        yva_d, yte_d = (yva == pos).astype(int), (yte == pos).astype(int)
        try:
            out["test_auc"] = round(float(roc_auc_score(yte_d, st)), 4)
            out["cnn_baseline"]["test_auc"] = round(
                float(roc_auc_score(yte_d, Pte_cnn[:, pos])), 4)
        except Exception as e:
            log("  [warn] AUC failed: %s: %s" % (type(e).__name__, e))

        # Both operating points are chosen on VALIDATION and then measured once on test.
        grid = np.arange(0.02, 0.99, 0.005)

        def pred_at(score, t):
            return np.where(score >= t, pos, neg)

        va = [accuracy_score(yva, pred_at(sv, t)) for t in grid]
        t_acc = float(grid[int(np.argmax(va))])

        # Screening: the highest threshold still catching >=MIN_SENS of the DISEASE on val.
        sens_v = np.array([((sv >= t)[yva_d == 1]).mean() for t in grid])
        ok = grid[sens_v >= MIN_SENS]
        t_scr = float(ok.max()) if len(ok) else float(grid[0])

        out["cnn_baseline"]["default_argmax"] = binary_rates(yte, Pte_cnn.argmax(1), pos)
        out["threshold_accuracy_optimal"] = {
            "threshold": round(t_acc, 3),
            "test_accuracy": round(float(accuracy_score(yte, pred_at(st, t_acc))), 4),
            **binary_rates(yte, pred_at(st, t_acc), pos)}
        out["threshold_screening_min_sens_%.2f" % MIN_SENS] = {
            "threshold": round(t_scr, 3),
            "test_accuracy": round(float(accuracy_score(yte, pred_at(st, t_scr))), 4),
            **binary_rates(yte, pred_at(st, t_scr), pos)}
        out["threshold_note"] = (
            "The accuracy-optimal point is for benchmark comparison only. On a screening "
            "task it can raise accuracy while catching FEWER diseased cases - measured here "
            "- so the screening point is the one a triage tool should serve.")
        for nm in ("threshold_accuracy_optimal",
                   "threshold_screening_min_sens_%.2f" % MIN_SENS):
            r = out[nm]
            log("[thr] %-34s t=%.3f  acc=%.4f  sens=%.4f  spec=%.4f  caught %s"
                % (nm, r["threshold"], r["test_accuracy"], r["sensitivity"],
                   r["specificity"], r["disease_caught"]))
        b = out["cnn_baseline"]["default_argmax"]
        log("[thr] %-34s t=0.500  acc=%.4f  sens=%.4f  spec=%.4f  caught %s"
            % ("cnn_argmax_baseline", base_test, b["sensitivity"], b["specificity"],
               b["disease_caught"]))

    log("\n[report]")
    log(classification_report(yte, pt.argmax(1), target_names=classes, zero_division=0))
    log("[RESULT] %s_hybrid  cnn_test=%.4f  hybrid_test=%.4f  delta=%+.4f  (%s + %s)"
        % (KEY, base_test, test_acc, test_acc - base_test, best["features"], best["head"]))

    p = os.path.join(MODEL_DIR, "%s_hybrid_metrics.json" % KEY)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log("wrote %s  (%.1fs)" % (p, time.time() - t0))


if __name__ == "__main__":
    main()
