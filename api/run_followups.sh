#!/usr/bin/env bash
# Ordered follow-up queue, cheapest-and-most-valuable first. Strictly serial: a 4 GB GPU
# cannot hold two of these at once.
#
#   1. oct_bin retry with AMP=0, only if the AMP run hit the nan guard
#   2. threshold tuning        ~10 min total, no retraining, likely the biggest win per minute
#   3. breast ensemble         5 seeds, 0.8782 -> ?
#   4. retina_bin ensemble     5 seeds, 0.8825 -> ?
#   5. oct multi-class AMP=0   last on purpose: published ceiling is ~0.78 and it is already
#                              at 0.775, so it is the least valuable hour in the queue
#   6. verification            re-measure v1 and v2 through one harness + leak check
set -u
API="c:/Users/so/Desktop/AI-Powered Radiology Hub/api"
PY="$API/venv_gpu/Scripts/python.exe"
LOG="$API/models/_retrain91.log"
cd "$API" || exit 1

# --- single-runner lock (added after session 2) -------------------------------------------
# Session 2 lost ~22 hours because this script and run_ensembles_after_sweep.sh both waited
# on the SAME marker line ("=== retrain_below91 end") and therefore both fired at the same
# instant on a 4 GB GPU / 8 GB box. Nearly every job died of CUDA OOM or MemoryError; the one
# member that survived took 94x longer than the same job run alone (77,268s vs 819s) and
# scored WORSE (0.8475 vs 0.8825). mkdir is atomic on every filesystem we care about, so this
# guarantees only one runner is ever alive no matter how many get launched.
LOCK="$API/models/.runner.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[lock] another runner already holds $LOCK — refusing to start."
  echo "[lock] Two runners at once is exactly what broke session 2. See TRAINING_LOG.md step 9."
  echo "[lock] If you are sure nothing is running: rmdir '$LOCK'"
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
echo "[lock] acquired $LOCK"

COMMON="WORKERS=0 TQDM_DISABLE=1 PYTHONIOENCODING=utf-8"

echo "[q] waiting for the sweep to end ..."
until grep -q "=== retrain_below91 end" "$LOG" 2>/dev/null; do sleep 20; done
echo "[q] sweep ended $(date '+%H:%M:%S')"

# --- 1. oct_bin retry without mixed precision --------------------------------------------
if grep -q "^\[RESULT\] oct_bin " "$LOG"; then
  echo "[q] oct_bin already succeeded under AMP — no retry needed"
else
  echo "[q] === oct_bin retry with AMP=0 $(date '+%H:%M:%S') ==="
  env $COMMON DATASET=octmnist BINARY=1 SIZE=128 EPOCHS=12 BATCH=64 MAX_TRAIN=40000 \
      VAL_MAX=3000 WARMUP=2 PATIENCE=5 DROPOUT=0.3 AMP=0 \
      "$PY" train_medmnist_v2.py 2>&1 | tee -a "$LOG"
fi

# --- 2. threshold tuning (no retraining) -------------------------------------------------
echo "[q] === threshold tuning $(date '+%H:%M:%S') ==="
"$PY" tune_threshold.py 2>&1 | tee -a "$LOG"

# --- 3. breast ensemble ------------------------------------------------------------------
echo "[q] === breast ensemble $(date '+%H:%M:%S') ==="
env $COMMON DATASET=breastmnist SEEDS=5 SIZE=224 EPOCHS=40 BATCH=32 WARMUP=4 \
    PATIENCE=12 DROPOUT=0.4 "$PY" train_ensemble.py

# --- 4. retina_bin ensemble --------------------------------------------------------------
echo "[q] === retina_bin ensemble $(date '+%H:%M:%S') ==="
env $COMMON DATASET=retinamnist BINARY=1 SEEDS=5 SIZE=224 EPOCHS=40 BATCH=32 WARMUP=4 \
    PATIENCE=12 DROPOUT=0.4 "$PY" train_ensemble.py

# --- 5. oct multi-class, no mixed precision ----------------------------------------------
echo "[q] === oct multi-class retry with AMP=0 $(date '+%H:%M:%S') ==="
env $COMMON DATASET=octmnist SIZE=128 EPOCHS=14 BATCH=64 MAX_TRAIN=40000 VAL_MAX=3000 \
    WARMUP=2 PATIENCE=5 DROPOUT=0.3 SUFFIX=_v2 AMP=0 \
    "$PY" train_medmnist_v2.py 2>&1 | tee -a "$LOG"

# --- 6. verification ---------------------------------------------------------------------
echo "[q] === verification $(date '+%H:%M:%S') ==="
VERIFY_DEVICE=cuda "$PY" verify_retrain_gains.py 2>&1 | tee -a "$LOG"

echo "FOLLOWUPS_ALL_DONE $(date '+%H:%M:%S')"
