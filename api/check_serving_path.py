# -*- coding: utf-8 -*-
"""
Does the HTTP endpoint return what the offline evaluation says it should?

This is the test the repo did not have, and its absence cost three separate serving bugs that
all shipped for months (TRAINING_LOG steps 54, 57, 59):

  - hflip TTA was measured into every published number and never applied at serve time;
  - val-tuned decision thresholds were written into checkpoints and never applied either;
  - `_read_image` converted every upload to greyscale, so `blood` served at 0.2441 against a
    published 0.9787 and `derma_bin` caught 10 of 392 malignancies instead of 310.

Every existing tool - verify_retrain_gains.py, fix_tta_selection.py, tune_threshold.py,
ensemble_archs.py - reads arrays straight out of `medmnist` and rebuilds the network itself.
Not one of them goes through `_read_image`, the multipart decode, or the predictor wrapper. So
all four agreed with each other and all four were blind to the path the user's request takes.

What this does: pull real images from the official test split, save them as PNG exactly as a
user would upload one, POST them to a running API, and compare the returned class and
probability against an offline forward pass in the model's OWN served configuration (arch,
input size, TTA flag, and promoted threshold). Any divergence is a serving bug by definition -
the two paths are supposed to be the same computation.

    python check_serving_path.py                       # every image model, 6 images each
    python check_serving_path.py --n 20 derma_bin      # one model, more images
    API=http://127.0.0.1:8000 python check_serving_path.py
"""
import os, io, sys, json, uuid, argparse, urllib.request
import numpy as np
import torch
import medmnist
from medmnist import INFO
from PIL import Image

from sklearn.metrics import confusion_matrix

from nets import build_medmnist_backbone, MEDMNIST_ARCHS
import preproc

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
API = os.environ.get("API", "http://127.0.0.1:8000")
TOL = float(os.environ.get("TOL", "0.01"))     # 1 percentage point on the reported probability
IM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# served key -> the MedMNIST dataset behind it
SERVED = [
    ("breast", "breastmnist"), ("derma", "dermamnist"), ("derma_bin", "dermamnist"),
    ("blood", "bloodmnist"), ("organc", "organcmnist"), ("path", "pathmnist"),
    ("oct", "octmnist"), ("oct_bin", "octmnist"),
    ("retina", "retinamnist"), ("retina_bin", "retinamnist"),
]


def served_checkpoints(key):
    """The checkpoint files main.py would actually load, in order - manifest first."""
    man = os.path.join(MODEL_DIR, f"{key}_ensemble.json")
    if os.path.exists(man):
        with open(man, encoding="utf-8") as f:
            return [os.path.join(MODEL_DIR, m) for m in json.load(f)["members"]], True
    v2 = os.path.join(MODEL_DIR, f"{key}_v2.pt")
    return ([v2] if os.path.exists(v2) else [os.path.join(MODEL_DIR, f"{key}.pt")]), False


def served_metrics(key, is_ens):
    for name in ([f"{key}_ens_metrics.json"] if is_ens else []) + \
                [f"{key}_v2_metrics.json", f"{key}_metrics.json"]:
        p = os.path.join(MODEL_DIR, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return {}


@torch.no_grad()
def offline(paths, pil_rgb):
    """One forward pass per member, in each member's own configuration, averaged.

    Deliberately re-implemented from a PIL image rather than from the medmnist array: the
    point is to reproduce what the server does to an uploaded FILE, including the RGB decode
    and the resize, not what the training harness does to a tensor.
    """
    acc, meta = None, None
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        arch, size = ck.get("arch", "resnet18"), ck.get("size", 64)
        if arch not in MEDMNIST_ARCHS:
            raise SystemExit(f"{p}: arch {arch} not buildable here")
        net, _ = build_medmnist_backbone(arch, num_classes=len(ck["classes"]),
                                         pretrained=False, dropout=ck.get("dropout", 0.0))
        net.eval(); net.load_state_dict(ck["state_dict"])
        # Fixed preprocessing before the resize, exactly as the server does it - ben_graham's
        # blur radius scales with image size, so doing it after would use a different sigma.
        im = preproc.apply_pil(pil_rgb.convert("RGB"), ck.get("preproc")).resize((size, size))
        x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)
        x = ((x - IM_MEAN) / IM_STD).unsqueeze(0)
        q = preproc.tta_average(net, x, preproc.views_for(ck)[0])[0]
        acc = q if acc is None else acc + q
        meta = meta or ck
        del net
    return (acc / len(paths)).numpy(), meta


def post(key, png_bytes):
    bd = "----" + uuid.uuid4().hex
    body = (f'--{bd}\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + png_bytes + f"\r\n--{bd}--\r\n".encode()
    r = urllib.request.Request(f"{API}/predict/{key}", data=body,
                               headers={"Content-Type": f"multipart/form-data; boundary={bd}"})
    return json.load(urllib.request.urlopen(r, timeout=300))


def run_full(args):
    """Score the entire official test split through the live endpoint.

    The per-image check below compares the API against `offline()` in this same file - which
    I also wrote. A mistake made in both would pass it, and that is precisely the failure
    documented in step 60: four tools agreed with each other and all four were blind to the
    same assumption. This mode compares against something neither of them produced - the
    accuracy recorded in the metrics file, measured months or hours earlier by the training
    script on raw arrays.

    Slow on purpose: one HTTP round trip per image. breast is 156 images (~40 s), retina 400.
    """
    for key, dataset in SERVED:
        if args.keys and key not in args.keys:
            continue
        paths, is_ens = served_checkpoints(key)
        if not all(os.path.exists(p) for p in paths):
            continue
        met = served_metrics(key, is_ens)
        pub = met.get("test_accuracy")
        classes = met.get("classes") or torch.load(paths[0], map_location="cpu",
                                                   weights_only=False)["classes"]
        ck0 = torch.load(paths[0], map_location="cpu", weights_only=False)
        size = ck0.get("size", 64)
        src = min([s for s in (64, 128, 224) if s >= size] or [224])
        ds = getattr(medmnist, INFO[dataset]["python_class"])(
            split="test", download=True, size=src, root=DATA_ROOT)
        y = ds.labels.astype(np.int64).reshape(-1)
        if ck0.get("binary_positive"):
            pos = set(ck0["binary_positive"])
            y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)
        pred = []
        for i in range(len(y)):
            a = ds.imgs[i]
            pil = Image.fromarray(a if a.ndim == 3 else np.repeat(a[..., None], 3, 2))
            buf = io.BytesIO(); pil.save(buf, format="PNG")
            pred.append(classes.index(post(key, buf.getvalue())["prediction_en"]))
        pred = np.array(pred)
        acc = float((pred == y).mean())
        # The published value is rounded to 4 dp, so compare at that precision - not with an
        # exact-equality tolerance, which reports a match as a mismatch.
        ok = pub is not None and abs(round(acc, 4) - pub) < 5e-5
        print("%-12s n=%-5d API=%.4f  published=%s  %s"
              % (key, len(y), acc, pub, "MATCH" if ok else "MISMATCH <-- investigate"))
        cm = confusion_matrix(y, pred)
        if len(classes) == 2:
            pi = 1 if ck0.get("binary_positive") else (0 if dataset == "breastmnist" else 1)
            print("             confusion %s   disease caught %d/%d"
                  % (cm.tolist(), cm[pi][pi], cm[pi].sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="images per model")
    ap.add_argument("--full", action="store_true",
                    help="push the WHOLE test split through HTTP and compare the "
                         "aggregate accuracy to the published metrics file")
    ap.add_argument("keys", nargs="*")
    args = ap.parse_args()

    if args.full:
        return run_full(args)

    rng = np.random.RandomState(0)
    total_bad, rows = 0, {}
    print(f"API={API}  tolerance={TOL:.3f} on the reported probability\n")
    print("%-12s %5s %8s %8s   %s" % ("model", "n", "class ok", "prob ok", "verdict"))
    print("-" * 78)
    for key, dataset in SERVED:
        if args.keys and key not in args.keys:
            continue
        paths, is_ens = served_checkpoints(key)
        if not all(os.path.exists(p) for p in paths):
            print("%-12s SKIP (checkpoint missing)" % key)
            continue
        met = served_metrics(key, is_ens)
        ck0 = torch.load(paths[0], map_location="cpu", weights_only=False)
        classes, size = ck0["classes"], ck0.get("size", 64)
        thr = None
        if ck0.get("binary_positive") and "threshold" in str(
                met.get("test_accuracy_source", "")).lower():
            thr = ck0.get("threshold")

        src = min([s for s in (64, 128, 224) if s >= size] or [224])
        ds = getattr(medmnist, INFO[dataset]["python_class"])(
            split="test", download=True, size=src, root=DATA_ROOT)
        idx = rng.choice(len(ds.imgs), size=min(args.n, len(ds.imgs)), replace=False)

        cls_ok = prob_ok = 0
        worst = 0.0
        for i in idx:
            arr = ds.imgs[int(i)]
            pil = Image.fromarray(arr if arr.ndim == 3 else np.repeat(arr[..., None], 3, 2))
            buf = io.BytesIO(); pil.save(buf, format="PNG")
            got = post(key, buf.getvalue())
            probs, _ = offline(paths, pil)
            if thr is not None and len(classes) == 2:
                top_i = 1 if float(probs[1]) >= thr else 0
            else:
                top_i = int(np.argmax(probs))
            cls_ok += (got["prediction_en"] == classes[top_i])
            api_p = {f["id"]: f["probability"] / 100.0 for f in got["findings"]}
            d = max(abs(api_p.get(c, 0.0) - float(probs[j])) for j, c in enumerate(classes))
            worst = max(worst, d)
            prob_ok += (d <= TOL)
        bad = len(idx) - min(cls_ok, prob_ok)
        total_bad += bad
        verdict = "MATCH" if bad == 0 else f"MISMATCH  worst prob delta {worst:.4f}"
        print("%-12s %5d %8s %8s   %s%s"
              % (key, len(idx), f"{cls_ok}/{len(idx)}", f"{prob_ok}/{len(idx)}", verdict,
                 "  [ensemble]" if is_ens else ""))
        rows[key] = {"n": int(len(idx)), "class_matches": int(cls_ok),
                     "prob_matches": int(prob_ok), "worst_prob_delta": round(float(worst), 5),
                     "ensemble": bool(is_ens), "tta": bool(ck0.get("tta", False)),
                     "threshold": thr}

    out = os.path.join(MODEL_DIR, "_serving_path_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\nwrote", out)
    if total_bad:
        print(f"\n[FAIL] {total_bad} image(s) disagree between the endpoint and an offline "
              f"forward pass in the served configuration. That is a serving bug, not noise.")
        sys.exit(1)
    print("\n[OK] the endpoint reproduces the offline computation on every image tested.")


if __name__ == "__main__":
    main()
