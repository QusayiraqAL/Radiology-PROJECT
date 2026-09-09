# -*- coding: utf-8 -*-
"""
Measure what the API's grayscale upload path costs each colour model.

The bug (found 2026-09-08, session 6). `main.py:_read_image` ends with

    return Image.open(io.BytesIO(raw)).convert("L")

so EVERY uploaded image is converted to greyscale before it reaches any model. The MedMNIST
predictor then calls `.convert("RGB")`, which only replicates that one grey channel three
times. Colour never reaches the network.

Six models were trained on colour and are served this way:

    derma / derma_bin   dermoscopy - pigment network colour is the diagnostic signal
    blood               stained smears - the stain IS the class information
    path                H&E histopathology - haematoxylin blue vs eosin pink, literally
    retina / retina_bin fundus - haemorrhages and exudates are identified by colour

Every published accuracy for them was measured on RGB. This script measures both, on the same
official test split, through the same code, and reports the difference. It changes nothing.

    python measure_grayscale_damage.py                 # all colour models
    python measure_grayscale_damage.py derma_bin       # one
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
from sklearn.metrics import accuracy_score, confusion_matrix

from nets import build_medmnist_backbone, MEDMNIST_ARCHS

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DATA_ROOT = os.path.join(HERE, "data", "medmnist")
DEVICE = os.environ.get("GRAY_DEVICE", "cpu")
BATCH = int(os.environ.get("GRAY_BATCH", "8"))
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# served key -> checkpoint file. Only models whose source data is RGB.
COLOUR_MODELS = [
    ("retina",     "retina_v2.pt",     "retinamnist"),
    ("retina_bin", "retina_bin.pt",    "retinamnist"),
    ("derma",      "derma_v2.pt",      "dermamnist"),
    ("derma_bin",  "derma_bin.pt",     "dermamnist"),
    ("blood",      "blood.pt",         "bloodmnist"),
    ("path",       "path.pt",          "pathmnist"),
]


class DS(Dataset):
    """`gray=True` reproduces the API path exactly: RGB -> L -> replicated back to 3 channels.

    PIL's L conversion is the ITU-R 601-2 luma transform, 0.299R + 0.587G + 0.114B. Doing the
    same arithmetic here rather than round-tripping through PIL keeps this measurement honest
    about what it is testing - and it is the identical operation.
    """
    def __init__(self, X, y, size, gray):
        self.X, self.y, self.gray = X, y, gray
        self.tf = transforms.Compose([
            transforms.ToPILImage(), transforms.Resize((size, size)),
            transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        im = self.X[i]
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        if self.gray:
            lum = (0.299 * im[..., 0] + 0.587 * im[..., 1] + 0.114 * im[..., 2])
            im = np.repeat(np.round(lum).clip(0, 255).astype(np.uint8)[..., None], 3, axis=-1)
        return self.tf(np.ascontiguousarray(im)), int(self.y[i])


@torch.no_grad()
def run(net, X, y, size, tta, gray):
    out = []
    for xb, _ in DataLoader(DS(X, y, size, gray), batch_size=BATCH, num_workers=0):
        xb = xb.to(DEVICE)
        p = torch.softmax(net(xb), 1)
        if tta:
            p = (p + torch.softmax(net(torch.flip(xb, dims=[3])), 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def main():
    only = sys.argv[1:]
    report = {}
    print("%-12s %-16s %9s %9s %9s   %s" % ("model", "arch", "RGB", "greyscale", "cost", "disease caught"))
    print("-" * 92)
    for key, ckpt_name, dataset in COLOUR_MODELS:
        if only and key not in only:
            continue
        cp = os.path.join(MODEL_DIR, ckpt_name)
        if not os.path.exists(cp):
            print("%-12s checkpoint missing (%s)" % (key, ckpt_name))
            continue
        ck = torch.load(cp, map_location=DEVICE, weights_only=False)
        arch = ck.get("arch", "resnet18")
        if arch not in MEDMNIST_ARCHS:
            print("%-12s SKIP arch=%s" % (key, arch))
            continue
        size, classes = ck.get("size", 64), ck["classes"]
        net, _ = build_medmnist_backbone(arch, num_classes=len(classes), pretrained=False,
                                         dropout=ck.get("dropout", 0.0))
        net = net.to(DEVICE).eval()
        net.load_state_dict(ck["state_dict"])

        DataClass = getattr(medmnist, INFO[dataset]["python_class"])
        src = min([s for s in (64, 128, 224) if s >= size] or [224])
        ds = DataClass(split="test", download=True, size=src, root=DATA_ROOT)
        X, y = ds.imgs, ds.labels.astype(np.int64).reshape(-1)
        bp = ck.get("binary_positive")
        if bp:
            pos = set(bp)
            y = np.array([1 if int(v) in pos else 0 for v in y], dtype=np.int64)

        tta = bool(ck.get("tta", False))
        thr = ck.get("threshold") if bp else None
        # Serve exactly as main.py would, including a promoted threshold.
        mp = os.path.join(MODEL_DIR, key + "_v2_metrics.json")
        if not os.path.exists(mp):
            mp = os.path.join(MODEL_DIR, key + "_metrics.json")
        met = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
        if "threshold" not in str(met.get("test_accuracy_source", "")).lower():
            thr = None

        def decide(p):
            if thr is not None and len(classes) == 2:
                return np.where(p[:, 1] >= thr, 1, 0)
            return p.argmax(1)

        p_rgb = run(net, X, y, size, tta, gray=False)
        p_gry = run(net, X, y, size, tta, gray=True)
        a_rgb = accuracy_score(y, decide(p_rgb))
        a_gry = accuracy_score(y, decide(p_gry))

        caught = ""
        if len(classes) == 2:
            pos_i = 1 if bp else (0 if dataset == "breastmnist" else 1)
            c_r = confusion_matrix(y, decide(p_rgb))[pos_i][pos_i]
            c_g = confusion_matrix(y, decide(p_gry))[pos_i][pos_i]
            tot = int((y == pos_i).sum())
            caught = "%d/%d -> %d/%d" % (c_r, tot, c_g, tot)
            report_extra = {"disease_caught_rgb": "%d/%d" % (c_r, tot),
                            "disease_caught_greyscale": "%d/%d" % (c_g, tot)}
        else:
            report_extra = {}

        print("%-12s %-16s %9.4f %9.4f %+9.4f   %s"
              % (key, arch, a_rgb, a_gry, a_gry - a_rgb, caught))
        report[key] = dict({"arch": arch, "n_test": int(len(y)), "tta": tta,
                            "threshold": thr, "accuracy_rgb": round(float(a_rgb), 4),
                            "accuracy_greyscale": round(float(a_gry), 4),
                            "cost": round(float(a_gry - a_rgb), 4)}, **report_extra)
        del net, X, p_rgb, p_gry

    out = os.path.join(MODEL_DIR, "_greyscale_damage.json")
    prev = {}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            prev = json.load(f)
    prev.update(report)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prev, f, ensure_ascii=False, indent=2)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
