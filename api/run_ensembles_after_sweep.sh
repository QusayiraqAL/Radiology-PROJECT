#!/usr/bin/env bash
# ==========================================================================================
# SUPERSEDED — do not run this alongside run_followups.sh.
#
# run_followups.sh already contains both jobs below (its steps 3 and 4) plus four more. This
# script was written first; when the fuller queue was written, this one was never retired,
# and because BOTH wait on the same marker line they both fired at once. That is the whole
# cause of the session-2 failure cascade (TRAINING_LOG.md, step 9).
#
# Kept for the record. The lock below now makes a double-run impossible, but the correct
# action is still: run run_followups.sh, not this.
# ==========================================================================================
# Waits for retrain_below91.py to finish, then runs seed-ensembles on the two models that
# landed closest to the 91% bar. Chained so the GPU is never idle and never contended:
# a 4 GB card cannot hold two of these runs at once.
#
#   breast_v2  0.8782  -> 5 seeds
#   retina_bin 0.8825  -> 5 seeds
#
# Both are small datasets (546 and 1080 training images) where a single seed's score swings
# by several points, which is exactly the case where averaging seeds is worth its compute.
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

echo "[chain] waiting for the sweep to finish ..."
until grep -q "=== retrain_below91 end" "$LOG" 2>/dev/null; do sleep 20; done
echo "[chain] sweep finished at $(date '+%H:%M:%S') — starting ensembles"

echo "[chain] === breast ensemble (5 seeds) ==="
DATASET=breastmnist SEEDS=5 SIZE=224 EPOCHS=40 BATCH=32 WARMUP=4 PATIENCE=12 \
  DROPOUT=0.4 WORKERS=0 TQDM_DISABLE=1 PYTHONIOENCODING=utf-8 \
  "$PY" train_ensemble.py
echo "[chain] breast ensemble rc=$?"

echo "[chain] === retina_bin ensemble (5 seeds) ==="
DATASET=retinamnist BINARY=1 SEEDS=5 SIZE=224 EPOCHS=40 BATCH=32 WARMUP=4 PATIENCE=12 \
  DROPOUT=0.4 WORKERS=0 TQDM_DISABLE=1 PYTHONIOENCODING=utf-8 \
  "$PY" train_ensemble.py
echo "[chain] retina_bin ensemble rc=$?"

echo "ENSEMBLES_ALL_DONE"
