#!/bin/bash
# The retraining plan with recipe "beta" (director's OK 2026-10-19; rule in PROGRESS.md, written before training),
# run unattended. Jobs go through scripts/job_queue.sh (one at a time, low priority, memory waits, resumes after
# memory stops); scripts/plan_gate.py applies the rule after each stage.
#
#   setsid nohup scripts/run_plan.sh > /dev/null 2>&1 &
#
# 1. A42 (+ validation; also the old B's validation scores, needed for the two-seed rule). A42 < 500: stop.
# 2. B42 with plain random colors. Can't drive (< 100): B switches to Slaoui's method and B42 is trained again
#    that way. If that can't drive either: stop.
# 3. A142, B142. Two-seed rule fails: stop.
# 4. A242, B242, the evaluations on the analysis tracks, the unchanged Step 3 tests (results/step3_beta),
#    and the speed check of every new agent.
# Ends by writing one line (when + why) to the done-file, once nothing of mine is running and memory is freed.
# Log: logs/plan.log. Gate decisions: results/beta_plan/gates.md. Results are committed after each stage.
# PLAN_DRY_RUN=<dir> (tests only): jobs and gates are replaced by stand-ins from <dir>, nothing is committed.
cd "$(dirname "$0")/.." || exit 1
DRY=${PLAN_DRY_RUN:-}
FLAG=${DRY:+$DRY/flag}; FLAG=${FLAG:-logs/training_done.flag}
LOG=${DRY:+$DRY/plan.log}; LOG=${LOG:-logs/plan.log}
A=beta
mkdir -p logs
p() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

commit_results() {
  [ -n "$DRY" ] && { p "(dry run) would commit: $1"; return; }
  git add -- results/ >> "$LOG" 2>&1
  git diff --cached --quiet -- results/ && return
  git commit -q -m "$1" -- results/ >> "$LOG" 2>&1 \
    && p "committed: $1" || p "commit failed: $1"
}

finish() {
  p "PLAN END: $1"
  commit_results "Retraining plan (recipe beta): $1"
  # Nothing of mine left running (the small memory guard may stay), then memory back near WSL's idle level.
  [ -n "$DRY" ] && { echo "$(date '+%Y-%m-%d %H:%M') - $1." > "$FLAG"; exit 0; }
  while ps -eo args | grep "\.venv/bin/python" | grep -v grep | grep -qv "scripts.memory_guard"; do sleep 30; done
  for i in $(seq 20); do
    used=$(free -m | awk '/^Mem:/ {print $3}')
    [ "$used" -lt 1600 ] && break
    sleep 30
  done
  echo "$(date '+%Y-%m-%d %H:%M') - $1. No training running; memory freed (WSL using ${used} MB)." > "$FLAG"
  p "done-file written"
  exit 0
}

queue() {
  if [ -n "$DRY" ]; then for j in "$@"; do p "(dry run) job: $j"; done; rc=$(cat "$DRY/queue_rc" 2>/dev/null || echo 0)
  else scripts/job_queue.sh "$@" > /dev/null; rc=$?; fi
  [ "$rc" -eq 0 ] || finish "stopped early, the job queue gave up or a job failed: $(tail -1 logs/job_queue.log | sed 's/^\[[^]]*\] //')"
}

# gate <name> [args]: 0 = passed, 10 = failed by the rule, anything else = the gate itself broke
gate() {
  if [ -n "$DRY" ]; then rc=$(grep "^$1 ${3:--} " "$DRY/gates" | awk '{print $NF}'); p "(dry run) gate $* -> $rc"
  else .venv/bin/python -m scripts.plan_gate "$@" >> "$LOG" 2>&1; rc=$?; fi
  [ $rc -eq 0 ] || [ $rc -eq 10 ] || finish "stopped early, the gate script failed (exit $rc) at gate $1"
  return $rc
}

train() { echo "scripts.train --agent $1 --seed $2 --recipe $3"; }
val() { echo "scripts.evaluate_agent --agent $1 --seeds $2 --recipe $3 --tracks validation --colors normal"; }

rm -f "$FLAG"
p "PLAN START (pid $$)"

# Stage 1
queue "$(train A 42 $A)" "$(val A 42 $A)" "$(val B '42 142' entropy)"
commit_results "Plan stage 1: A seed 42 trained with recipe beta, validation scores"
gate a42 || finish "stopped by the rule: A seed 42 scored under 500 on validation, so the recipe or my code doesn't work here (results/beta_plan/gates.md)"

# Stage 2
B=beta
queue "$(train B 42 $B)" "$(val B 42 $B)"
commit_results "Plan stage 2: B seed 42 trained with recipe beta (plain random colors), validation scores"
if ! gate b42 --recipe-b $B; then
  B=beta_slaoui
  p "B seed 42 can't drive with plain random colors: B switches to Slaoui's method ($B), as the rule says"
  queue "$(train B 42 $B)" "$(val B 42 $B)"
  commit_results "Plan stage 2b: B seed 42 trained with Slaoui's method, validation scores"
  gate b42 --recipe-b $B || finish "stopped by the rule: B seed 42 can't drive with plain random colors or with Slaoui's method (results/beta_plan/gates.md)"
fi
p "Agent B recipe: $B"

# Stage 3
queue "$(train A 142 $A)" "$(val A 142 $A)" "$(train B 142 $B)" "$(val B 142 $B)"
commit_results "Plan stage 3: A and B seed 142 trained, validation scores"
gate two_seeds --recipe-b $B || finish "stopped by the rule: the two-seed check (seeds 42 + 142) failed (results/beta_plan/gates.md)"

# Stage 4
queue "$(train A 242 $A)" "$(val A 242 $A)" "$(train B 242 $B)" "$(val B 242 $B)" \
      "scripts.evaluate_agent --agent A --seeds 42 142 242 --recipe $A --tracks analysis --colors normal unseen" \
      "scripts.evaluate_agent --agent B --seeds 42 142 242 --recipe $B --tracks analysis --colors normal unseen" \
      "scripts.compare_agents --out-dir results/step3_beta --recipe-a $A --recipe-b $B"
commit_results "Plan stage 4: seeds 242 trained; analysis-track evaluations and the Step 3 color tests (results/step3_beta)"
speed=()
for agent in A B; do
  r=$A; [ $agent = B ] && r=$B
  for seed in 42 142 242; do
    speed+=("scripts.diagnose_speed --agent $agent --seed $seed --recipe $r --tracks validation --colors normal --out results/beta_plan/speed/${agent}_seed${seed}")
  done
done
queue "${speed[@]}"
finish "finished: all training in the plan is done (A: recipe $A, B: recipe $B, seeds 42, 142, 242), plus the evaluations, the Step 3 color tests (results/step3_beta) and the speed checks"
