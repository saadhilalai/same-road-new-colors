"""
What-if test, no training: could the trained driver handle more speed? Plays the 100 validation tracks with the
driver's own gas multiplied by a factor (clipped to the allowed 0..1). Steering and brake are untouched.
Factor 1.0 must reproduce the normal evaluation exactly (a built-in check).

    python -m scripts.diagnose_gas_scaling --agent A --seed 42 --factors 1.0 1.5 2.0 3.0 --out results/step2/gas_scaling

Reading the result (written down before running it, in PROGRESS.md):
  more gas -> higher score / more laps without many more crashes: the steering copes; the learned throttle is the limit.
  more gas -> many more off-track races: the driver can't control higher speeds; it needs to learn braking/anticipation.
"""

import argparse
import json

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

import config
from racecar.evaluate import evaluate, summarize
from racecar.train_utils import build_eval_vec_env


class ScaledGas:
    """Wraps a trained model: same decisions, gas multiplied by `factor`."""

    def __init__(self, model, factor):
        self.model, self.factor = model, factor

    def predict(self, obs, deterministic=True):
        action, state = self.model.predict(obs, deterministic=deterministic)
        action = action.copy()
        action[:, 1] = np.clip(action[:, 1] * self.factor, 0.0, 1.0)
        return action, state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipe", default=config.RECIPE)
    p.add_argument("--factors", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run = config.run_name(args.agent, args.seed, args.recipe)
    model = PPO.load(config.RUNS_DIR / run / "checkpoints" / "model_final.zip", device=config.DEVICE)
    reference = pd.read_csv(config.RESULTS_DIR / "evals" / run / "validation_normal.csv")
    out = config.PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    venv = build_eval_vec_env(config.EVAL_N_ENVS, config.CONTINUOUS_ACTIONS, config.GRAYSCALE)
    rows = []
    for f in args.factors:
        df = evaluate(ScaledGas(model, f), [(t, None) for t in config.VALIDATION_TRACKS], venv=venv)
        df.to_csv(out / f"validation_normal_gas_x{f:g}.csv", index=False)
        timeout = ~df.lap_completed & ~df.off_track & (df.frames >= 1000)
        row = {"gas_factor": f, **summarize(df), "out_of_time_pct": round(100 * timeout.mean(), 1),
               "out_of_time_with_90pct_covered": int((timeout & (df.track_covered_pct >= 90)).sum())}
        if f == 1.0:
            same = np.allclose(df.sort_values("track_seed").score.values, reference.sort_values("track_seed").score.values)
            row["matches_normal_evaluation"] = bool(same)
        rows.append(row)
        print(json.dumps(row), flush=True)
    venv.close()
    table = pd.DataFrame(rows)
    (out / "summary.md").write_text(f"# Gas-scaling what-if: {run}, 100 validation tracks, normal colors\n\n"
                                    + table.to_markdown(index=False) + "\n")
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
