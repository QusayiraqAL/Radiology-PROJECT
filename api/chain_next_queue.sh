#!/usr/bin/env bash
# Wait for the current runner to release the lock, then start the next queue.
#
# run_sweep.sh refuses to start while models/.runner.lock exists (the session-4 fix for the
# collision that cost 21.8 hours). That is correct, but it also means a finished queue leaves
# the GPU idle until someone notices. This waits on the lock rather than on a log line - the
# session-2 failure was two scripts watching the same log line and both firing at once, and a
# lock cannot be raced that way.
set -u
API="$(cd "$(dirname "$0")" && pwd)"
QUEUE="${1:?usage: chain_next_queue.sh <queue_file>}"
LOCK="$API/models/.runner.lock"

. "$API/lock_lib.sh"
# Wait on a HELD lock, not merely a present one: a lock abandoned by a killed runner would
# otherwise make this wait forever, which is the stall it is supposed to recover from.
while lock_is_held "$LOCK"; do sleep 30; done
echo "[chain] lock released - starting $(basename "$QUEUE")"
exec bash "$API/run_sweep.sh" "$QUEUE"
