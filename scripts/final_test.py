"""
Step 6: the final test, run once on the hidden test tracks, exactly as written in results/final_test/TEST_PLAN.md.

    python -m scripts.final_test

Refuses to start unless the plan and this script are committed to git unchanged. Refuses to run again once finished.
After an interruption (e.g. a memory stop) it finishes only the missing runs: same races, same AI, same results.
Writes results/evals/<run name>/test_<colors>.csv, runs/<run name>/actions/test_<colors>.npz (every action, for
exact replays), the untrained baseline, and the Step 3 analysis in results/final_test/.
"""

import json
import subprocess
import sys
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO

import config
from racecar.evaluate import evaluate, summarize

OUT = config.RESULTS_DIR / "final_test"
PLAN = OUT / "TEST_PLAN.md"
STARTED, FINISHED = OUT / "started.json", OUT / "finished.json"
BASELINE = dict(agent="A", seed=42, checkpoint=0)


def git(*args):
    return subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True)


def committed_unchanged(path):
    rel = str(path.relative_to(config.PROJECT_ROOT))
    return git("ls-files", "--error-unmatch", rel).returncode == 0 and git("diff", "--quiet", "HEAD", "--", rel).returncode == 0


def races(colors):
    return [(t, None if colors == "normal" else t) for t in config.TEST_TRACKS]


def save(df, actions, run, name):
    out = config.RESULTS_DIR / "evals" / run
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (config.RUNS_DIR / run / "actions").mkdir(exist_ok=True)
    np.savez_compressed(config.RUNS_DIR / run / "actions" / f"{name}.npz",
                        **{f"{t}_{c}": a for (t, c), a in actions.items()})


def main():
    for path in (PLAN, config.PROJECT_ROOT / "scripts" / "final_test.py"):
        if not committed_unchanged(path):
            sys.exit(f"refused: {path.name} must be committed to git unchanged before the final test")
    if FINISHED.exists():
        sys.exit("refused: the final test has already been run (results/final_test/finished.json)")
    if STARTED.exists():
        print(f"resuming after an interruption: {STARTED.read_text().strip()}", flush=True)
    else:
        STARTED.write_text(json.dumps({"started": datetime.now().isoformat(timespec="seconds"),
                                       "commit": git("rev-parse", "HEAD").stdout.strip()}) + "\n")

    jobs = [(agent, seed, colors) for agent in sorted(config.AGENTS) for seed in config.TRAINING_SEEDS
            for colors in ("normal", "unseen")]
    for agent, seed, colors in jobs:
        run = config.run_name(agent, seed, config.RECIPE)
        if (config.RESULTS_DIR / "evals" / run / f"test_{colors}.csv").exists():
            continue
        model = PPO.load(config.RUNS_DIR / run / "checkpoints" / "model_final.zip", device=config.DEVICE)
        df, actions = evaluate(model, races(colors), record_actions=True)
        df.insert(0, "agent", agent)
        df.insert(1, "seed", seed)
        df.insert(2, "checkpoint", "final")
        df.insert(3, "recipe", config.RECIPE)
        save(df, actions, run, f"test_{colors}")
        print(f"done: {run} test {colors} ({len(df)} races)", flush=True)   # no scores printed before the end

    run = config.run_name(BASELINE["agent"], BASELINE["seed"], config.RECIPE)
    name = f"test_normal_ckpt{BASELINE['checkpoint']}_sampled"
    if not (config.RESULTS_DIR / "evals" / run / f"{name}.csv").exists():
        model = PPO.load(config.RUNS_DIR / run / "checkpoints" / f"model_{BASELINE['checkpoint']}.zip", device=config.DEVICE)
        df, actions = evaluate(model, races("normal"), record_actions=True, sampled=True)
        df.insert(0, "agent", "untrained")
        df.insert(1, "seed", BASELINE["seed"])
        df.insert(2, "checkpoint", BASELINE["checkpoint"])
        df.insert(3, "recipe", config.RECIPE)
        save(df, actions, run, name)
        print(f"done: untrained baseline ({len(df)} races)", flush=True)

    rc = subprocess.run([sys.executable, "-m", "scripts.compare_agents", "--tracks", "test", "--out-dir",
                         "results/final_test", "--title", "Final test"], cwd=config.PROJECT_ROOT).returncode
    if rc != 0:
        sys.exit(f"the analysis failed (exit {rc}); the evaluations are saved, rerun only the analysis")
    import pandas as pd
    baseline = summarize(pd.read_csv(config.RESULTS_DIR / "evals" / run / f"{name}.csv"))
    (OUT / "untrained_baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    FINISHED.write_text(json.dumps({"finished": datetime.now().isoformat(timespec="seconds")}) + "\n")
    print("final test finished", flush=True)


if __name__ == "__main__":
    main()
