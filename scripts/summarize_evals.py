"""
Summary table for evaluations: one row per seed, plus the mean over seeds.

    python -m scripts.summarize_evals --agent A --name analysis_normal --out results/step2/agentA_analysis_normal.md

"Mean over seeds" = average of the three per-seed means (each seed is one independent training run).
"""

import argparse

import pandas as pd

import config
from racecar.evaluate import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--name", required=True, help="e.g. analysis_normal, validation_normal")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = []
    for seed in config.TRAINING_SEEDS:
        path = config.RESULTS_DIR / "evals" / f"{args.agent}_seed{seed}" / f"{args.name}.csv"
        if path.exists():
            rows.append({"seed": seed, **summarize(pd.read_csv(path))})
    table = pd.DataFrame(rows)
    if len(table) > 1:
        numeric = table.drop(columns=["seed", "races", "solved_900"]).mean().round(1)
        table.loc[len(table)] = {"seed": "mean of seeds", "races": table["races"].sum(),
                                 "solved_900": bool(numeric["mean_score"] >= config.SOLVED_SCORE), **numeric}
    md = table.to_markdown(index=False)
    print(md)
    if args.out:
        with open(args.out, "w") as f:
            f.write(f"Agent {args.agent}, {args.name}\n\n{md}\n")


if __name__ == "__main__":
    main()
