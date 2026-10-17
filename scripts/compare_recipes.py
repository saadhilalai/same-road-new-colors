"""
Recipe trial (Step 2): compare recipes trained with the same agent and seed, on the same 100 validation tracks.

    python -m scripts.compare_recipes --agent A --seed 42 --recipes entropy kl_limit \
        --out results/step2/recipe_trial.md

Applies the selection rule written in PROGRESS.md before any result existed:
  winner = higher mean validation score of the final copy; paired Wilcoxon signed-rank test reported alongside;
  if no recipe reaches config.RECIPE_TRIAL_MIN_SCORE, there is no winner (stop and ask the director).
"""

import argparse
import json

import pandas as pd
from scipy.stats import wilcoxon

import config
from racecar.evaluate import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipes", nargs=2, required=True)
    p.add_argument("--name", default="validation_normal")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dfs = {r: pd.read_csv(config.RESULTS_DIR / "evals" / config.run_name(args.agent, args.seed, r) / f"{args.name}.csv")
           for r in args.recipes}
    table = pd.DataFrame([{"recipe": r, "change": json.dumps(config.RECIPE_FIXES[r]), **summarize(df)}
                          for r, df in dfs.items()])

    a, b = args.recipes
    paired = dfs[a][["track_seed", "score"]].merge(dfs[b][["track_seed", "score"]], on="track_seed", suffixes=(f"_{a}", f"_{b}"))
    diff = paired[f"score_{a}"] - paired[f"score_{b}"]
    test = wilcoxon(paired[f"score_{a}"], paired[f"score_{b}"])
    means = {r: dfs[r]["score"].mean() for r in args.recipes}
    best = max(means, key=means.get)
    winner = best if means[best] >= config.RECIPE_TRIAL_MIN_SCORE else None

    lines = [
        f"# Recipe trial: agent {args.agent}, seed {args.seed}, final copies, 100 validation tracks (normal colors)",
        "",
        table.to_markdown(index=False),
        "",
        f"Paired comparison on the same {len(paired)} tracks ({a} minus {b}):",
        f"- mean difference {diff.mean():+.1f} points; {a} better on {(diff > 0).sum()} tracks, "
        f"{b} better on {(diff < 0).sum()}, tied on {(diff == 0).sum()}",
        f"- Wilcoxon signed-rank test: statistic {test.statistic:.1f}, p = {test.pvalue:.4g} "
        f"({'significant' if test.pvalue < 0.05 else 'NOT significant'} at the 5% level)",
        "",
        (f"**Winner by the pre-written rule: `{winner}`** (higher mean; minimum needed {config.RECIPE_TRIAL_MIN_SCORE})."
         if winner else
         f"**No winner:** neither recipe reached a mean of {config.RECIPE_TRIAL_MIN_SCORE}. Stop and ask the director."),
    ]
    text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    print(json.dumps({"winner": winner, "p_value": test.pvalue}))


if __name__ == "__main__":
    main()
