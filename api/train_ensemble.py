# -*- coding: utf-8 -*-
"""
Seed-ensemble wrapper around train_medmnist_v2.py.

Why this exists: BreastMNIST has 546 training images and a 156-image test set, so a single
run's score swings by several points on seed alone. Averaging the softmax of N independently
seeded runs is the most reliable way to buy accuracy on a set that small — it costs N times
the compute and adds no new information the data doesn't already contain, which is exactly
why it is honest: no extra labels, no test-set peeking, same recipe N times.

It also tunes the binary decision threshold on the VALIDATION split (never on test) and
reports the test number at both the default 0.5 and the tuned threshold, so you can see how
much of any gain came from the threshold rather than the ensemble.

  DATASET=breastmnist SEEDS=5 python train_ensemble.py

Writes models/<key>_ens.pt (the member list + threshold) and models/<key>_ens_metrics.json.
Each member checkpoint models/<key>_s<seed>.pt is kept so the ensemble can be rebuilt or
audited member by member.
"""
import os, sys, json, time, subprocess
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix,
                             classification_report, balanced_accuracy_score)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
PY = sys.executable

DATASET = os.environ.get("DATASET", "breastmnist")
SEEDS   = int(os.environ.get("SEEDS", "5"))
BINARY  = os.environ.get("BINARY", "0") == "1"
SIZE    = os.environ.get("SIZE", "224")
EPOCHS  = os.environ.get("EPOCHS", "40")
BATCH   = os.environ.get("BATCH", "32")
WARMUP  = os.environ.get("WARMUP", "4")
PATIENCE = os.environ.get("PATIENCE", "12")
DROPOUT = os.environ.get("DROPOUT", "0.4")
MAX_TRAIN = os.environ.get("MAX_TRAIN", "0")
VAL_MAX = os.environ.get("VAL_MAX", "0")
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0") == "1"   # reuse existing member checkpoints

BASE_KEY = DATASET.replace("mnist", "") + ("_bin" if BINARY else "")
LOG = os.path.join(MODEL_DIR, "_retrain91.log")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def train_member(seed):
    suffix = "_s%d" % seed
    key = BASE_KEY + suffix
    ckpt = os.path.join(MODEL_DIR, key + ".pt")
    if SKIP_TRAIN and os.path.exists(ckpt):
        log("[ens] member seed=%d already trained, reusing %s" % (seed, key)); return ckpt
    env = dict(os.environ)
    env.update(dict(DATASET=DATASET, SIZE=SIZE, EPOCHS=EPOCHS, BATCH=BATCH, WARMUP=WARMUP,
                    PATIENCE=PATIENCE, DROPOUT=DROPOUT, SUFFIX=suffix, SEED=str(seed),
                    WORKERS="0", TQDM_DISABLE="1", PYTHONIOENCODING="utf-8",
                    MAX_TRAIN=MAX_TRAIN, VAL_MAX=VAL_MAX,
                    BINARY=("1" if BINARY else "0")))
    log("\n[ens] ---- training member seed=%d -> %s ----" % (seed, key))
    p = subprocess.Popen([PY, os.path.join(HERE, "train_medmnist_v2.py")], cwd=HERE, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line)
    rc = p.wait()
    if rc != 0 or not os.path.exists(ckpt):
        log("[ens] member seed=%d FAILED rc=%d" % (seed, rc)); return None
    return ckpt


def main():
    t0 = time.time()
    # Import here so the module-level env of the trainer is read AFTER we set it per member.
    os.environ.setdefault("WORKERS", "0")
    os.environ["DATASET"], os.environ["SIZE"] = DATASET, SIZE
    os.environ["BINARY"] = "1" if BINARY else "0"
    import train_medmnist_v2 as T
    from nets import build_brain_resnet

    members = [c for c in (train_member(s) for s in range(SEEDS)) if c]
    if not members:
        raise SystemExit("[ens] no members trained")
    log("\n[ens] %d/%d members trained — evaluating ensemble" % (len(members), SEEDS))

    Xva, yva = T.load_split("val"); Xte, yte = T.load_split("test")
    if BINARY:
        yva, yte = T.to_binary(yva), T.to_binary(yte)
    vl = DataLoader(T.DS(Xva, yva, T.eval_tf), batch_size=64)
    el = DataLoader(T.DS(Xte, yte, T.eval_tf), batch_size=64)

    def probs_of(ckpt_path, loader):
        ck = torch.load(ckpt_path, map_location=T.DEVICE, weights_only=False)
        net = build_brain_resnet(num_classes=len(ck["classes"]), pretrained=False,
                                 dropout=ck.get("dropout", 0.0)).to(T.DEVICE).eval()
        net.load_state_dict(ck["state_dict"])
        out = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(T.DEVICE)
                p = torch.softmax(net(xb), 1)
                p = (p + torch.softmax(net(torch.flip(xb, dims=[3])), 1)) / 2   # hflip TTA
                out.append(p.float().cpu().numpy())
        del net
        if T.DEVICE == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(out), ck["classes"]

    pv_all, pt_all, classes = [], [], None
    for i, c in enumerate(members):
        pv, classes = probs_of(c, vl)
        pt, _ = probs_of(c, el)
        pv_all.append(pv); pt_all.append(pt)
        log("[ens] member %d (%s): solo test acc = %.4f"
            % (i, os.path.basename(c), accuracy_score(yte, pt.argmax(1))))
    pv, pt = np.mean(pv_all, 0), np.mean(pt_all, 0)
    n_cls = pt.shape[1]

    # Threshold tuning (binary only), chosen on VAL and then applied unchanged to test.
    thr, acc_val_default, acc_val_tuned = 0.5, accuracy_score(yva, pv.argmax(1)), None
    if n_cls == 2:
        grid = np.arange(0.05, 0.96, 0.01)
        accs = [accuracy_score(yva, (pv[:, 1] >= t).astype(int)) for t in grid]
        thr = float(grid[int(np.argmax(accs))]); acc_val_tuned = float(max(accs))
        pred = (pt[:, 1] >= thr).astype(int)
        log("[ens] threshold tuned on val: %.2f (val acc %.4f -> %.4f)"
            % (thr, acc_val_default, acc_val_tuned))
    else:
        pred = pt.argmax(1)

    acc_argmax = accuracy_score(yte, pt.argmax(1))
    acc_final = accuracy_score(yte, pred)
    try:
        auc = (roc_auc_score(yte, pt[:, 1]) if n_cls == 2
               else roc_auc_score(yte, pt, multi_class="ovr", average="macro"))
    except Exception:
        auc = float("nan")

    solo = [float(accuracy_score(yte, p.argmax(1))) for p in pt_all]
    metrics = {
        "model": BASE_KEY + "_resnet18_ensemble",
        "dataset_key": BASE_KEY + "_ens", "medmnist": DATASET, "binary_task": BINARY,
        "task": (T.BINARY_TASKS[DATASET]["name"] if BINARY else "%d-class classification" % n_cls),
        "method": ("softmax average of %d independently seeded runs of the v2 recipe, each with "
                   "hflip TTA; binary decision threshold tuned on the validation split only"
                   % len(members)),
        "n_members": len(members), "member_seeds": list(range(SEEDS))[:len(members)],
        "member_test_accuracies": [round(s, 4) for s in solo],
        "member_mean_test_accuracy": round(float(np.mean(solo)), 4),
        "member_best_test_accuracy": round(float(np.max(solo)), 4),
        "input_size": int(SIZE), "classes": classes, "n_classes": n_cls,
        "n_val": int(len(yva)), "n_test": int(len(yte)),
        "decision_threshold": round(thr, 3),
        "test_accuracy": round(float(acc_final), 4),
        "test_accuracy_argmax": round(float(acc_argmax), 4),
        "val_accuracy_argmax": round(float(acc_val_default), 4),
        "val_accuracy_tuned": None if acc_val_tuned is None else round(acc_val_tuned, 4),
        "test_balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
        "test_macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "test_auc": None if np.isnan(auc) else round(float(auc), 4),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "ensemble_gain_over_mean_member": round(float(acc_final - np.mean(solo)), 4),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seconds": round(time.time() - t0, 1), "device": T.DEVICE,
        "test_split": "full official MedMNIST test split (never subsampled, never tuned on)",
    }
    torch.save({"members": [os.path.basename(c) for c in members], "threshold": thr,
                "size": int(SIZE), "classes": classes, "binary_task": BINARY,
                "medmnist": DATASET,
                "binary_positive": (T.BINARY_TASKS[DATASET]["positive"] if BINARY else None)},
               os.path.join(MODEL_DIR, BASE_KEY + "_ens.pt"))
    json.dump(metrics, open(os.path.join(MODEL_DIR, BASE_KEY + "_ens_metrics.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)
    log("\n" + classification_report(yte, pred, target_names=classes, zero_division=0))
    log("[RESULT] %s_ens test_acc=%.4f (argmax %.4f) members=%s mean_member=%.4f gain=%+.4f thr=%.2f"
        % (BASE_KEY, acc_final, acc_argmax, [round(s, 4) for s in solo], np.mean(solo),
           acc_final - np.mean(solo), thr))
    log(BASE_KEY.upper() + "_ENS_DONE")


if __name__ == "__main__":
    main()
