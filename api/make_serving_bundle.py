# -*- coding: utf-8 -*-
"""
Pack the exact set of files main.py needs to SERVE, and nothing else.

Why this exists. models/ is 1.9 GB, but a serving process never touches most of it. It
loads one checkpoint per registered model, and main.py always prefers the v2 retrain and
only falls back to v1 when v2 is absent (main.py:442-445) - so shipping both doubles the
payload for a file that will never be opened. Superseded experiments (retina_bin_s3/s4 from
the collided ensemble run, retina_ord, the _ar_v1_backup tree) are pure dead weight too.

Measured on this repo: 1.9 GB -> ~1.4 GB, and the difference is entirely files the server
would have ignored. That matters because the destination is a Colab session that has to
receive this over a network before it can answer a single request.

What is NOT bundled, on purpose:
  - `chest` has no local weights at all - torchxrayvision downloads densenet121-res224-all
    on first use, so it needs internet on the far side, not a file here.
  - MedMNIST quiz data: main.py downloads it on demand into api/data/.
  - v1 checkpoints, when a v2 exists. Serving them is not a fallback anyone wants; if v2 is
    missing from the bundle that is a packaging bug, and a silent v1 fallback would hide it.

  python make_serving_bundle.py                    # -> radiology_serving_bundle.zip
  python make_serving_bundle.py --out D:/some.zip
"""
import argparse
import glob
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")

# Mirrors MEDMNIST_MODELS in main.py:301. Kept as a literal rather than imported because
# importing main.py loads every model into memory - the opposite of what a packager wants.
MEDMNIST_KEYS = ["breast", "derma", "blood", "organc", "path",
                 "oct", "retina", "retina_bin", "derma_bin", "oct_bin"]

# (preferred, fallback) - same "v2 if present else v1" rule main.py applies at load time.
SINGLE_MODELS = [
    ("pneumonia_v2.pt", "pneumonia_xray.pt"),        # main.py:141-142
    ("brain_tumor_mri_v2.pt", "brain_tumor_mri.pt"),  # main.py:230-231
]

# Whole directories the Arabic + English text systems load from (ar_service.py:86-96).
# _ar_v1_backup is 161 MB of superseded weights that nothing imports.
DIR_TREES = [("ar", {"_ar_v1_backup"}), ("text", set())]

# The server's own source. Bundling it instead of cloning from GitHub is deliberate: the
# weights and the code that loads them then travel as ONE artifact and cannot drift apart,
# and the far side needs no credentials for a repo that may not be public. It is ~400 KB
# against 1.4 GB of weights, so the robustness is effectively free.
SOURCE_GLOBS = ["*.py", "requirements.txt"]

# main.py:487 serves this page at "/" when it sits next to api/. Including it means the
# tunnel URL opens the whole app on any device, instead of only answering JSON.
ROOT_FILES = ["Radiology Hub.html"]


def human(n):
    return "%.1f MB" % (n / 1e6) if n < 1e9 else "%.2f GB" % (n / 1e9)


def collect():
    """Return (list of (abs_path, arcname), list of warnings). Nothing is written yet."""
    picked, warn = [], []

    def add(rel):
        p = os.path.join(MODEL_DIR, rel)
        if os.path.exists(p):
            picked.append((p, "api/models/" + rel.replace("\\", "/")))
            return True
        return False

    for key in MEDMNIST_KEYS:
        # Same preference order as main.py, so the bundle serves what the local box serves.
        if not (add("%s_v2.pt" % key) or add("%s.pt" % key)):
            warn.append("no checkpoint for MedMNIST model '%s' - it will show as unavailable" % key)

    for preferred, fallback in SINGLE_MODELS:
        if not (add(preferred) or add(fallback)):
            warn.append("missing %s (and its %s fallback)" % (preferred, fallback))

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

    # Metrics and logs are the evidence behind every number the API reports in /models.
    # They are kilobytes, and a server that serves numbers without them is lying by omission.
    for fn in sorted(os.listdir(MODEL_DIR)):
        if fn.endswith("_metrics.json") or fn.endswith(".json") or fn.endswith(".log"):
            add(fn)

    for pattern in SOURCE_GLOBS:
        for full in sorted(glob.glob(os.path.join(HERE, pattern))):
            picked.append((full, "api/" + os.path.basename(full)))

    root = os.path.dirname(HERE)
    for fn in ROOT_FILES:
        full = os.path.join(root, fn)
        if os.path.exists(full):
            picked.append((full, fn))
        else:
            warn.append("missing %s - the tunnel will answer JSON but serve no UI" % fn)

    seen, unique = set(), []
    for p, arc in picked:
        if arc not in seen:
            seen.add(arc)
            unique.append((p, arc))
    return unique, warn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "radiology_serving_bundle.zip"))
    ap.add_argument("--dry-run", action="store_true", help="list what would be packed, write nothing")
    args = ap.parse_args()

    files, warn = collect()
    total = sum(os.path.getsize(p) for p, _ in files)
    full = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(MODEL_DIR) for f in fs)

    print("bundling %d files | %s of %s in models/ (%.0f%% left behind)"
          % (len(files), human(total), human(full), 100 * (1 - total / max(full, 1))))
    for p, arc in sorted(files, key=lambda t: -os.path.getsize(t[0]))[:12]:
        print("   %-46s %s" % (arc, human(os.path.getsize(p))))
    if len(files) > 12:
        print("   ... and %d smaller files" % (len(files) - 12))

    for w in warn:
        print("[warn]", w)

    if args.dry_run:
        print("\ndry run - nothing written")
        return 0

    out = os.path.abspath(args.out)
    # ZIP_STORED, not DEFLATE: .pt files are already-compressed tensor blobs, so deflating
    # them costs minutes of CPU to save low single-digit percent. Measured, not assumed.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for i, (p, arc) in enumerate(files, 1):
            z.write(p, arc)
            if i % 25 == 0 or i == len(files):
                print("  packed %d/%d" % (i, len(files)), flush=True)

    print("\nwrote %s (%s)" % (out, human(os.path.getsize(out))))
    print("Upload this ONE file to Google Drive, then run Serve_API_Colab.ipynb.")
    if warn:
        print("[!] %d warning(s) above - the bundle is usable but incomplete." % len(warn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
