"""
Training health check: is learning stable? Compares recipes trained with the same agent and seed.

    python -m scripts.plot_training_health --agent A --seed 42 --recipes zoo entropy kl_limit \
        --out results/step2/A_seed42_training_health.png

Four small charts sharing the same x-axis (AI decisions in training):
  1. Validation score (20 tracks, normal colors) at every checkpoint.
  2. Off track: % of training races that ended by leaving the playing area (last 100 races).
  3. How much each training round changed the driver (PPO's "approx KL"; higher = bigger change).
  4. How random the driver's actions still are (PPO's exploration "std"; lower = less trying new things).
"""

import argparse
import glob

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import config
from racecar import plot_style

plot_style.apply()
import matplotlib.pyplot as plt

# All lines are the same agent, so recipes are told apart by lightness + line style (plus the legend).
RECIPE_STYLES = {
    "zoo": dict(color=plot_style.MUTED, linestyle="-", label="Zoo recipe (pilot)"),
    "entropy": dict(color=plot_style.SEQUENTIAL_BLUES[2], linestyle="-", label="fix: entropy bonus 0.01"),
    "kl_limit": dict(color=plot_style.SEQUENTIAL_BLUES[4], linestyle="--", label="fix: update-size limit (KL 0.05)"),
}


def tb_scalars(run_dir):
    """All TensorBoard sessions of a run (a resumed run has several), joined into one series per tag."""
    out = {}
    for path in sorted(glob.glob(str(run_dir / "tb" / "*" / "events*"))):
        ev = EventAccumulator(path)
        ev.Reload()
        for tag in ("train/approx_kl", "train/std"):
            if tag in ev.Tags()["scalars"]:
                s = pd.Series({e.step: e.value for e in ev.Scalars(tag)})
                key = tag.split("/")[-1]
                out[key] = pd.concat([out[key], s]) if key in out else s
    return {k: v[~v.index.duplicated(keep="last")].sort_index() for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipes", nargs="+", default=["zoo"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for recipe in args.recipes:
        run_dir = config.RUNS_DIR / config.run_name(args.agent, args.seed, recipe)
        if not (run_dir / "validation.csv").exists():
            continue
        style = RECIPE_STYLES[recipe]
        tr = pd.read_csv(run_dir / "train_episodes.csv").sort_values("timesteps")
        va = pd.read_csv(run_dir / "validation.csv").groupby("timesteps")["score"].mean()
        tb = tb_scalars(run_dir)
        axes[0].plot(va.index / 1e6, va.values, marker="o", markersize=5, linewidth=1.6, **style)
        axes[1].plot(tr["timesteps"] / 1e6, 100 * tr["off_track"].astype(float).rolling(100, min_periods=20).mean(),
                     linewidth=1.6, **style)
        axes[2].plot(tb["approx_kl"].index / 1e6, tb["approx_kl"].rolling(8, min_periods=1).mean(), linewidth=1.6, **style)
        axes[3].plot(tb["std"].index / 1e6, tb["std"], linewidth=1.6, **style)

    ax = axes[0]
    ax.axhline(config.SOLVED_SCORE, color=plot_style.MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0.005, config.SOLVED_SCORE + 15, "solved (900)", color=plot_style.INK_SECONDARY, fontsize=9,
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Agent {args.agent}, seed {args.seed}: validation score "
                 f"({len(config.VALIDATION_DURING_TRAINING)} tracks, normal colors)", loc="left")

    axes[1].set_ylabel("% of races")
    axes[1].set_title("Left the playing area (training races, last 100)", loc="left")

    ax = axes[2]
    ax.axhspan(0.01, 0.05, color=plot_style.GRID, zorder=0)
    ax.text(0.005, 0.03, "usual PPO size (0.01-0.05)", color=plot_style.INK_SECONDARY, fontsize=9,
            va="center", transform=ax.get_yaxis_transform())
    ax.set_ylabel("approx KL")
    ax.set_title("How much each training round changed the driver", loc="left")

    ax = axes[3]
    ax.set_ylabel("exploration std")
    ax.set_title("How much random 'trying things' is left in its actions", loc="left")
    ax.set_xlabel("AI decisions in training (millions)")
    ax.set_xlim(0, config.TOTAL_TIMESTEPS / 1e6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
