"""
The judging rule of the retraining plan with recipe "beta" (written in PROGRESS.md before training, 2026-10-19).
Run by scripts/run_plan.sh after each stage. Validation tracks only (100 tracks, normal colors), final copies;
the scores come from scripts.evaluate_agent. Unseen colors are not looked at here.

    python -m scripts.plan_gate a42                       # A seed 42 at least PLAN_A42_MIN?
    python -m scripts.plan_gate b42 --recipe-b beta       # B seed 42 at least PLAN_B42_MIN ("can drive")?
    python -m scripts.plan_gate two_seeds --recipe-b beta # the two-seed rule (seeds 42 + 142)

Exit code 0 = passed, 10 = failed (run_plan.sh then switches B or stops, as the rule says), anything else = error.
Every decision is appended to results/beta_plan/gates.md.
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
from racecar.evaluate import summarize

OUT = config.RESULTS_DIR / "beta_plan"
PASSED, FAILED = 0, 10


def scores(agent, seed, recipe):
    """Per-track scores (index = track) on the 100 validation tracks, normal colors."""
    path = config.RESULTS_DIR / "evals" / config.run_name(agent, seed, recipe) / "validation_normal.csv"
    df = pd.read_csv(path)
    assert sorted(df["track_seed"]) == config.VALIDATION_TRACKS and (df["color_seed"] == -1).all(), path
    return df.set_index("track_seed")["score"], df


def seed_line(agent, seed, recipe):
    s = summarize(scores(agent, seed, recipe)[1])
    return (f"{agent} seed {seed} ({recipe}): mean {s['mean_score']} +/- {s['std_score']}, laps {s['lap_completed_pct']}%, "
            f"off track {s['off_track_pct']}%, pass mark {config.PILOT_PASS_SCORE}: "
            f"{'reached' if s['mean_score'] >= config.PILOT_PASS_SCORE else 'not reached'}")


def single_seed(agent, recipe, minimum, name):
    mean = scores(agent, 42, recipe)[0].mean()
    ok = mean >= minimum
    return ok, [seed_line(agent, 42, recipe), f"Rule: {name} needs a mean of at least {minimum}: "
                f"{mean:.1f} -> {'PASSED' if ok else 'FAILED'}"]


def two_seeds(recipe_b):
    seeds = (42, 142)
    old = config.PLAN_OLD_RECIPE
    per_track = lambda agent, recipe: pd.concat([scores(agent, s, recipe)[0] for s in seeds], axis=1).mean(axis=1)
    lines = [seed_line(a, s, r) for a, r in (("A", config.RECIPE), ("B", recipe_b)) for s in seeds]

    a_new, a_old = per_track("A", config.RECIPE).mean(), per_track("A", old).mean()
    a_ok = a_new >= a_old - config.PLAN_A_MAX_DROP
    lines.append(f"A: two-seed mean {a_new:.1f} vs old recipe '{old}' {a_old:.1f} (allowed: at most "
                 f"{config.PLAN_A_MAX_DROP} lower, so at least {a_old - config.PLAN_A_MAX_DROP:.1f}) -> "
                 f"{'PASSED' if a_ok else 'FAILED'}")

    b_new, b_old = per_track("B", recipe_b), per_track("B", old)
    diff = (b_new - b_old).loc[config.VALIDATION_TRACKS].to_numpy()
    p = float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
    b_min_ok = b_new.mean() >= config.PLAN_B_TWO_SEED_MIN
    b_sig_ok = diff.mean() > 0 and p < config.PLAN_ALPHA
    lines.append(f"B: two-seed mean {b_new.mean():.1f} (needs at least {config.PLAN_B_TWO_SEED_MIN}) -> "
                 f"{'PASSED' if b_min_ok else 'FAILED'}")
    lines.append(f"B vs old recipe '{old}' (two-seed mean {b_old.mean():.1f}): mean difference {diff.mean():+.1f}, better on "
                 f"{int((diff > 0).sum())} of 100 tracks, worse on {int((diff < 0).sum())}; paired Wilcoxon (two-sided) "
                 f"p = {p:.3g} (needs p < {config.PLAN_ALPHA} and a positive difference) -> {'PASSED' if b_sig_ok else 'FAILED'}")
    ok = a_ok and b_min_ok and b_sig_ok
    lines.append(f"Two-seed rule: {'PASSED, the new recipe is adopted' if ok else 'FAILED, stop and report'}")
    return ok, lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("gate", choices=["a42", "b42", "two_seeds"])
    p.add_argument("--recipe-b", choices=sorted(config.BETA_RECIPES), default=config.RECIPE)
    args = p.parse_args()

    if args.gate == "a42":
        ok, lines = single_seed("A", config.RECIPE, config.PLAN_A42_MIN, "A seed 42 (recipe or code broken if not)")
    elif args.gate == "b42":
        ok, lines = single_seed("B", args.recipe_b, config.PLAN_B42_MIN, f"B seed 42 'can drive' ({args.recipe_b})")
    else:
        ok, lines = two_seeds(args.recipe_b)

    OUT.mkdir(parents=True, exist_ok=True)
    md = OUT / "gates.md"
    if not md.exists():
        md.write_text("# Retraining plan with recipe 'beta': gate decisions\n\nRule: PROGRESS.md, written before "
                      "training. Validation tracks, normal colors, final copies.\n")
    with open(md, "a") as f:
        f.write(f"\n## {args.gate} (B recipe: {args.recipe_b}), {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write("".join(f"- {line}\n" for line in lines))
    with open(OUT / "gates.jsonl", "a") as f:
        f.write(json.dumps({"gate": args.gate, "recipe_b": args.recipe_b, "passed": ok,
                            "time": datetime.now().isoformat(timespec="seconds"), "lines": lines}) + "\n")
    print("\n".join(lines), flush=True)
    sys.exit(PASSED if ok else FAILED)


if __name__ == "__main__":
    main()
