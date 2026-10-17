"""
Evaluate trained agents on validation or analysis tracks, with normal or unseen colors.
The hidden test tracks are NOT available here (Step 6 has its own script).

    python -m scripts.evaluate_agent --agent A --seeds 42 --tracks validation --colors normal
    python -m scripts.evaluate_agent --agent A --tracks analysis --colors normal unseen

Writes results/evals/<agent>_seed<seed>/<tracks>_<colors>.csv (one row per race)
and runs/<agent>_seed<seed>/actions/<tracks>_<colors>.npz (every action, for exact replays).
"""

import argparse
import json

import numpy as np
from stable_baselines3 import PPO

import config
from racecar.evaluate import evaluate, summarize

TRACK_SETS = {"validation": config.VALIDATION_TRACKS, "analysis": config.ANALYSIS_TRACKS}


def races(tracks, colors):
    """Normal colors: color seed None. Unseen colors: color seed = track number (see config.py)."""
    return [(t, None if colors == "normal" else t) for t in tracks]


def evaluate_run(agent, seed, track_set, colors, checkpoint="final"):
    run_dir = config.RUNS_DIR / f"{agent}_seed{seed}"
    model = PPO.load(run_dir / "checkpoints" / f"model_{checkpoint}.zip", device=config.DEVICE)
    df, actions = evaluate(model, races(TRACK_SETS[track_set], colors), record_actions=True)
    df.insert(0, "agent", agent)
    df.insert(1, "seed", seed)
    df.insert(2, "checkpoint", checkpoint)
    name = f"{track_set}_{colors}" + ("" if checkpoint == "final" else f"_ckpt{checkpoint}")
    out = config.RESULTS_DIR / "evals" / f"{agent}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (run_dir / "actions").mkdir(exist_ok=True)
    np.savez_compressed(run_dir / "actions" / f"{name}.npz", **{f"{t}_{c}": a for (t, c), a in actions.items()})
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=sorted(config.AGENTS), required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=config.TRAINING_SEEDS)
    p.add_argument("--tracks", choices=sorted(TRACK_SETS), required=True)
    p.add_argument("--colors", choices=["normal", "unseen"], nargs="+", default=["normal"])
    p.add_argument("--checkpoint", default="final")
    args = p.parse_args()

    for seed in args.seeds:
        for colors in args.colors:
            df = evaluate_run(args.agent, seed, args.tracks, colors, args.checkpoint)
            print(json.dumps({"agent": args.agent, "seed": seed, "tracks": args.tracks, "colors": colors,
                              "checkpoint": args.checkpoint, **summarize(df)}), flush=True)


if __name__ == "__main__":
    main()
