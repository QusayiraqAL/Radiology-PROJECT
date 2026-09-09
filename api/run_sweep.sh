#!/usr/bin/env bash
# Session 6 general sweep runner.  Usage:  bash run_sweep.sh <queue_file>
#
# The queue file holds one run per line:   <tag>  <ENV=VAL> <ENV=VAL> ...
# Blank lines and #-comments are skipped. Runs execute STRICTLY in order, one at a time.
#
# Why the lock: session 2 lost 21.8 hours because two runners shared a 4 GB card. mkdir is
# atomic, so a second copy of this script exits instead of colliding (session 4 fix).
#
# Constants pinned for every row, so the only difference between rows is the lever tested:
#   AMP=0     fp16 -> loss=nan on this GTX 1650 (measured twice: sessions 1 and 3)
#   WORKERS=0 spawned Windows workers each import torch (~1 GB) on an 8 GB box
#   SEED=0    same seed as the v2 baselines
# SIZE and EPOCHS default to the v2 baseline values but a queue line may override them.
set -u
API="$(cd "$(dirname "$0")" && pwd)"
QUEUE="${1:?usage: run_sweep.sh <queue_file>}"
LOG="$API/models/_arch_sweep.log"
PY="$API/venv_gpu/Scripts/python.exe"

LOCK="$API/models/.runner.lock"
. "$API/lock_lib.sh"
if ! mkdir "$LOCK" 2>/dev/null; then
  if lock_is_held "$LOCK"; then
    echo "[lock] another runner already holds $LOCK (pid $(lock_owner_pid "$LOCK" || true)) — refusing to start."
    exit 1
  fi
  # Owner is provably gone. Reclaim rather than sit idle - this is the 15-hour stall of
  # 2026-09-08 (TRAINING_LOG step 88), and the reason is logged so it is never silent.
  echo "[lock] $LOCK is stale: recorded pid '$(lock_owner_pid "$LOCK" || true)' is gone. Reclaiming." | tee -a "$LOG"
  rm -rf "$LOCK"
  mkdir "$LOCK" 2>/dev/null || { echo "[lock] lost the race to reclaim $LOCK — refusing to start."; exit 1; }
fi
lock_write_owner "$LOCK"
# rm -rf, not rmdir: the lock now holds a pid file, so rmdir would fail and leave it behind.
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT

echo "=== sweep $(basename "$QUEUE") start $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
while read -r tag rest; do
  case "$tag" in ''|'#'*) continue ;; esac
  echo ""                                                    >> "$LOG"
  echo "======== $tag  $(date '+%Y-%m-%d %H:%M:%S') ========" >> "$LOG"
  echo "[cmd] $rest"                                         >> "$LOG"
  ( cd "$API" && env AMP=0 WORKERS=0 SEED=0 SIZE=224 EPOCHS=30 $rest "$PY" train_medmnist_v2.py ) \
      >> "$LOG" 2>&1
  echo "[exit] $tag rc=$?"                                   >> "$LOG"
done < "$QUEUE"
echo "=== sweep $(basename "$QUEUE") end $(date '+%Y-%m-%d %H:%M:%S') ===" >> "$LOG"
