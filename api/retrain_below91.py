# -*- coding: utf-8 -*-
"""
Driver for the "everything under 91%" retrain pass.

Runs train_medmnist_v2.py once per job below, in order, each in its own subprocess (so a
CUDA OOM or a bad dataset download kills one job, not the whole sweep), and appends every
line of output to models/_retrain91.log so the run is reproducible from the log alone.

  python retrain_below91.py            # all jobs
  python retrain_below91.py breast     # only jobs whose key starts with 'breast'

Job list — why each one is here (v1 test accuracy in brackets):
  breast   [0.808]  binary US. Small set (546) — the two-stage fine-tune at 224 is the lever.
  oct      [0.775]  v1 saw only 20k of 97,477 train images. Full data + 128px is the lever.
  oct_bin           normal-vs-disease screening question.
  derma    [0.722]  7-class HAM10000, heavily imbalanced. 224px is the lever.
  derma_bin         malignant-vs-benign — the question a triage tool answers.
  retina   [0.495]  5-grade DR. Published ResNet-18 ceiling is ~0.52, so treat >0.50 as the
                    real target here and let the binary head carry the clinical use.
  retina_bin        referable DR (grade >= 2) — the DR-screening question.
"""
import os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "models", "_retrain91.log")
PY = sys.executable

# Hardware this was tuned for: GTX 1650 Max-Q (4 GB VRAM) on an 8 GB-RAM laptop.
# WORKERS=0 everywhere on purpose: Windows spawns DataLoader workers as fresh processes that
# each import torch (~1 GB), and with ~3 GB free RAM two of them wedged the run at 0% GPU.
# One process is slower per epoch but it actually finishes.
COMMON = dict(WORKERS="0", TQDM_DISABLE="1", PYTHONIOENCODING="utf-8")

# (job key, env overrides)
JOBS = [
    ("breast",     dict(DATASET="breastmnist", SIZE="224", EPOCHS="40", BATCH="32",
                        WARMUP="4", PATIENCE="12", DROPOUT="0.4", SUFFIX="_v2")),
    ("retina",     dict(DATASET="retinamnist", SIZE="224", EPOCHS="40", BATCH="32",
                        WARMUP="4", PATIENCE="12", DROPOUT="0.4", SUFFIX="_v2")),
    ("retina_bin", dict(DATASET="retinamnist", SIZE="224", EPOCHS="40", BATCH="32",
                        WARMUP="4", PATIENCE="12", DROPOUT="0.4", BINARY="1")),
    ("derma",      dict(DATASET="dermamnist", SIZE="224", EPOCHS="22", BATCH="32",
                        WARMUP="3", PATIENCE="7", DROPOUT="0.4", SUFFIX="_v2")),
    ("derma_bin",  dict(DATASET="dermamnist", SIZE="224", EPOCHS="22", BATCH="32",
                        WARMUP="3", PATIENCE="7", DROPOUT="0.4", BINARY="1")),
    # oct: 128px (not 224) and 40k of 97,477 train images — still 2x what v1 saw — because a
    # full-res full-data pass is ~4 h per run on this GPU. VAL_MAX caps the 10,832-image val
    # split used for checkpoint selection; the TEST split stays complete.
    ("oct",        dict(DATASET="octmnist", SIZE="128", EPOCHS="14", BATCH="64", MAX_TRAIN="40000",
                        VAL_MAX="3000", WARMUP="2", PATIENCE="5", DROPOUT="0.3", SUFFIX="_v2")),
    ("oct_bin",    dict(DATASET="octmnist", SIZE="128", EPOCHS="12", BATCH="64", MAX_TRAIN="40000",
                        VAL_MAX="3000", WARMUP="2", PATIENCE="5", DROPOUT="0.3", BINARY="1")),
]


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    jobs = [j for j in JOBS if not only or any(j[0].startswith(o) for o in only)]
    with open(LOG, "a", encoding="utf-8") as log:
        log.write("\n%s\n=== retrain_below91 start %s | %d job(s) ===\n"
                  % ("=" * 78, time.strftime("%Y-%m-%d %H:%M:%S"), len(jobs)))
        log.flush()
        for key, env_over in jobs:
            hdr = "\n----- JOB %s | %s -----" % (key, time.strftime("%H:%M:%S"))
            print(hdr, flush=True); log.write(hdr + "\n"); log.flush()
            env = dict(os.environ); env.update(COMMON); env.update(env_over)
            t0 = time.time()
            p = subprocess.Popen([PY, os.path.join(HERE, "train_medmnist_v2.py")], cwd=HERE, env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in p.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                log.write(line); log.flush()
            rc = p.wait()
            tail = "----- JOB %s rc=%d in %.1fs -----\n" % (key, rc, time.time() - t0)
            print(tail, flush=True); log.write(tail); log.flush()
        log.write("=== retrain_below91 end %s ===\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("RETRAIN91_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
