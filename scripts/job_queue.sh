#!/bin/bash
# Runs my jobs one at a time (director's memory rule: only one fits next to the PC's other apps), at low priority.
# Before each start it waits until memory has recovered (scripts/wait_for_memory.py: 1 min steady for a first
# start, 10 min after a memory stop). If the memory guard stops
# a job, it waits and starts it again: a training resumes exactly from its last checkpoint; an evaluation simply
# reruns (its results are deterministic). Any other failure ends the queue.
#
#   setsid nohup scripts/job_queue.sh "scripts.train --agent B --seed 142 --recipe entropy" \
#                                     "scripts.evaluate_agent --agent B --seeds 142 --tracks analysis --colors normal unseen" &
#
# Log: logs/job_queue.log. Training output goes to runs/<run>/train.log, other jobs to logs/<job>.log.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
MAX_TRIES=8
NICE=$(.venv/bin/python -c "import config; print(config.NICE_LEVEL)")
renice -n "$NICE" $$ > /dev/null   # the queue itself also runs at low priority
q() { echo "[$(date '+%F %T')] $*" >> logs/job_queue.log; echo "$*"; }

for job in "$@"; do
  if [[ $job == scripts.train* ]]; then
    run=$(.venv/bin/python -c "
import sys, config
a = sys.argv[1:]
arg = lambda k: a[a.index(k) + 1] if k in a else None
print(config.run_name(arg('--agent'), int(arg('--seed')), arg('--recipe')))" $job)
    mkdir -p "runs/$run"; out="runs/$run/train.log"
  else
    out="logs/$(echo "${job#scripts.}" | sed -E 's/--//g; s/[^A-Za-z0-9]+/_/g').log"
  fi
  ok=0
  for try in $(seq $MAX_TRIES); do
    if [ "$try" -eq 1 ]; then .venv/bin/python -m scripts.wait_for_memory --first-start; else .venv/bin/python -m scripts.wait_for_memory; fi
    stops=$(grep -c "STOP:" logs/memory_guard.log)
    q "START (try $try): $job"
    setsid -w nice -n "$NICE" .venv/bin/python -m $job >> "$out" 2>&1 < /dev/null; rc=$?
    if [ $rc -eq 0 ]; then q "DONE: $job"; ok=1; break; fi
    if [ "$(grep -c 'STOP:' logs/memory_guard.log)" -gt "$stops" ]; then
      q "MEMORY STOP (rc $rc): $job; restarting once memory recovers"; continue
    fi
    q "FAILED (rc $rc): $job; see $out. Queue ends."; exit 1
  done
  [ $ok -eq 1 ] || { q "GAVE UP after $MAX_TRIES tries: $job. Queue ends."; exit 1; }
done
q "QUEUE FINISHED"
