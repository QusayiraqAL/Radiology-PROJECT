# -*- coding: utf-8 -*-
"""
Stage exactly what main.py needs to SERVE into a directory ready to push to a Hugging
Face Space, and nothing else.

Why this exists. models/ is 3.6 GB, but a serving process never touches most of it. It
loads one checkpoint per registered model, and main.py always prefers the v2 retrain and
only falls back to v1 when v2 is absent - so shipping both doubles the payload for a file
that will never be opened. Superseded experiments (the collided retina_bin ensemble seeds,
retina_ord, the _ar_v1_backup tree, every k-fold and arch-sweep checkpoint) are dead weight
too. Measured: 3.6 GB -> 1.30 GB, and the difference is entirely files the server ignores.

This descends from the Colab bundler deleted in session 6 (commit 2d5f8a1, removed in
session 6 step 33). The reason it is back is not the reason it left: serving from Colab was
a detour around a VRAM limit that turned out not to exist, while this is a host that stays
up when the laptop does not. The file selection logic is the part worth keeping, and it is
unchanged - it still mirrors main.py's own preference order.

What is NOT staged, on purpose:
  - `chest` has no local weights at all - torchxrayvision downloads densenet121-res224-all
    on first use, so it needs network on the far side, not a file here.
  - MedMNIST quiz data: main.py downloads it on demand into api/data/.
  - v1 checkpoints when a v2 exists. A silent v1 fallback would hide a packaging bug.

  python make_space_bundle.py --dry-run          # list what would be staged
  python make_space_bundle.py --out ../.space    # write the staging tree
"""
import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

# Mirrors MEDMNIST_MODELS in main.py. Kept as a literal rather than imported because
# importing main.py loads every model into memory - the opposite of what a packager wants.
MEDMNIST_KEYS = ["breast", "derma", "blood", "organc", "path",
                 "oct", "retina", "retina_bin", "derma_bin", "oct_bin"]

# (preferred, fallback) - same "v2 if present else v1" rule main.py applies at load time.
SINGLE_MODELS = [
    ("pneumonia_v2.pt", "pneumonia_xray.pt"),
    ("brain_tumor_mri_v2.pt", "brain_tumor_mri.pt"),
]

# Whole directories the Arabic + English text systems load from (ar_service.py).
# _ar_v1_backup is superseded weights that nothing imports.
DIR_TREES = [("ar", {"_ar_v1_backup"}), ("text", set())]

# The server's own source. Staging it rather than cloning from GitHub is deliberate: the
# weights and the code that loads them travel as ONE artifact and cannot drift apart.
SOURCE_GLOBS = ["*.py", "requirements.txt"]

# main.py serves this page at "/" when it sits next to api/.
ROOT_FILES = ["Radiology Hub.html"]

# Ensemble manifests name their members; whatever they point at has to come along.
ENSEMBLE_SUFFIX = "_ensemble.json"


def human(n):
    return "%.1f MB" % (n / 1e6) if n < 1e9 else "%.2f GB" % (n / 1e9)


def collect():
    """Return (list of (abs_path, arcname), warnings). Nothing is written yet."""
    picked, warn = [], []
    seen = set()

    def add(rel):
        p = os.path.join(MODEL_DIR, rel)
        arc = "api/models/" + rel.replace("\\", "/")
        if os.path.exists(p) and arc not in seen:
            seen.add(arc)
            picked.append((p, arc))
            return True
        return os.path.exists(p)

    for key in MEDMNIST_KEYS:
        if not (add("%s_v2.pt" % key) or add("%s.pt" % key)):
            warn.append("no checkpoint for MedMNIST model '%s' - it will show unavailable" % key)

    for preferred, fallback in SINGLE_MODELS:
        if not (add(preferred) or add(fallback)):
            warn.append("missing %s (and its %s fallback)" % (preferred, fallback))

    # Ensembles: main.py reads <key>_ensemble.json and loads every member it names.
    # Missing a member turns a promoted ensemble back into a single model silently.
    import json
    for fn in sorted(os.listdir(MODEL_DIR)):
        if not fn.endswith(ENSEMBLE_SUFFIX):
            continue
        add(fn)
        try:
            man = json.load(open(os.path.join(MODEL_DIR, fn), encoding="utf-8"))
        except Exception as e:
            warn.append("could not read %s (%s) - its members may be missing" % (fn, e))
            continue
        # main.py reads members as bare filenames under models/. Tolerate a dict form
        # too, so a future manifest that carries per-member metadata still packs.
        for m in (man.get("members") or []):
            ck = m if isinstance(m, str) else (m.get("checkpoint") or m.get("file"))
            if ck and not add(os.path.basename(ck)):
                warn.append("%s names member %s, which is not in models/" % (fn, ck))

    for sub, skip in DIR_TREES:
        root = os.path.join(MODEL_DIR, sub)
        if not os.path.isdir(root):
            warn.append("missing models/%s/ - that whole subsystem will be unavailable" % sub)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                add(os.path.relpath(full, MODEL_DIR))

    # Metrics and logs are the evidence behind every number /models reports. They are
    # kilobytes, and a server that serves numbers without them is lying by omission.
    for fn in sorted(os.listdir(MODEL_DIR)):
        if fn.endswith(".json") or fn.endswith(".log"):
            add(fn)

    for pattern in SOURCE_GLOBS:
        for full in sorted(glob.glob(os.path.join(HERE, pattern))):
            arc = "api/" + os.path.basename(full)
            if arc not in seen:
                seen.add(arc)
                picked.append((full, arc))

    # space/ holds the files that exist only for the Space: the Dockerfile, its pinned
    # requirements, and the README whose YAML front matter is how HF is told to build with
    # Docker on port 7860. They land at the staging root, not under api/.
    space_dir = os.path.join(os.path.dirname(HERE), "space")
    if os.path.isdir(space_dir):
        for fn in sorted(os.listdir(space_dir)):
            full = os.path.join(space_dir, fn)
            if os.path.isfile(full):
                picked.append((full, fn))
    else:
        warn.append("missing space/ - the Space will have no Dockerfile and will not build")

    root = os.path.dirname(HERE)
    for fn in ROOT_FILES:
        full = os.path.join(root, fn)
        if os.path.exists(full):
            picked.append((full, fn))
        else:
            warn.append("missing %s - the Space will answer JSON but serve no UI" % fn)

    return picked, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), ".space"))
    ap.add_argument("--dry-run", action="store_true", help="list what would be staged")
    args = ap.parse_args()

    files, warn = collect()
    total = sum(os.path.getsize(p) for p, _ in files)
    every = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(MODEL_DIR) for f in fs)
    print("staging %d files | %s of %s in models/ (%d%% left behind)"
          % (len(files), human(total), human(every), 100 - round(100.0 * total / every)))
    for p, arc in sorted(files, key=lambda x: -os.path.getsize(x[0]))[:10]:
        print("   %-48s %s" % (arc, human(os.path.getsize(p))))
    if len(files) > 10:
        print("   ... and %d smaller files" % (len(files) - 10))
    for w in warn:
        print("   !! " + w)

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    out = os.path.abspath(args.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    for p, arc in files:
        dst = os.path.join(out, arc)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
    print("\nstaged -> %s" % out)
    return 1 if warn else 0


if __name__ == "__main__":
    sys.exit(main())
