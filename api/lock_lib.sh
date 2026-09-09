# Shared runner-lock helpers, sourced by run_sweep.sh and chain_next_queue.sh.
#
# models/.runner.lock is an atomic mkdir: session 2 lost 21.8 hours to two runners sharing one
# 4 GB card, and the session-4 fix was to make a second copy refuse to start rather than collide.
# That part still holds and is not being loosened here.
#
# What it did not handle is a lock that outlives its owner. run_sweep.sh clears the lock from an
# EXIT trap, and the trap never fires when the process is killed without a chance to run - a
# shutdown, or SIGKILL. That happened on 2026-09-08 13:16 and left the card idle for 15 hours
# with no error anywhere, because nothing could tell a held lock from an abandoned one
# (TRAINING_LOG step 88). So the owner now records its pid inside the lock dir and these helpers
# read it back.
#
# The rule both callers follow: only a lock whose recorded owner is provably gone gets reclaimed.
# No pid file, an unreadable pid, or a live pid all count as HELD. Reclaiming too eagerly would
# reintroduce the exact collision the lock exists to prevent, so every doubt resolves that way -
# a stalled queue costs idle hours, a collision costs the runs themselves.
#
# Note on pids: "$$" here is the MSYS pid, not the Windows pid, so it is only comparable from
# inside git-bash. Both callers are git-bash scripts, so that holds. MSYS pids can be reused;
# a reused pid makes lock_is_held say "held", which is the safe direction.

lock_write_owner() {          # $1 = lock dir
  echo "$$" > "$1/pid"
}

lock_owner_pid() {            # $1 = lock dir -> recorded pid, or empty
  cat "$1/pid" 2>/dev/null || true
}

lock_is_held() {              # $1 = lock dir. rc 0 = held (or unknown), rc 1 = absent/abandoned.
  [ -d "$1" ] || return 1
  local owner
  owner="$(lock_owner_pid "$1")"
  # Empty or non-numeric: a pid we cannot ask about is not a pid we may declare dead. "kill -0
  # garbage" fails the same way "kill -0 <dead pid>" does, so without this digit check a corrupt
  # pid file would read as abandoned and get reclaimed - the one direction that costs runs.
  case "$owner" in ''|*[!0-9]*) return 0 ;; esac
  kill -0 "$owner" 2>/dev/null
}
