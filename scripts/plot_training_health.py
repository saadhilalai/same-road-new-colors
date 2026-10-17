"""
Training health check for one run: is learning stable?

    python -m scripts.plot_training_health --agent A --seed 42 --out results/step2/A_seed42_training_health.png

Four small charts sharing the same x-axis (AI decisions in training):
  1. Score: training races (mean of last 100) and validation checks (20 tracks).
  2. Off track: % of training races that ended by leaving the playing area.
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


def tb_scalars(run_dir):
    ev = EventAccumulator(sorted(glob.glob(str(run_dir / "tb" / "*" / "events*")))[-1])
    ev.Reload()
    return {t.split("/")[-1]: pd.Series({e.step: e.value for e in ev.Scalars(t)}).sort_index()
            for t in ("train/approx_kl", "train/std", "train/clip_fraction")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    run_dir = config.RUNS_DIR / f"{args.agent}_seed{args.seed}"
    color = plot_style.AGENT_COLORS[args.agent]
    tr = pd.read_csv(run_dir / "train_episodes.csv").sort_values("timesteps")
    va = pd.read_csv(run_dir / "validation.csv").groupby("timesteps").agg(
        score=("score", "mean"), off=("off_track", "mean"))
    tb = tb_scalars(run_dir)
    x_tr = tr["timesteps"] / 1e6

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    ax = axes[0]
    ax.plot(x_tr, tr["score"].rolling(100, min_periods=20).mean(), color=color, linewidth=1.6,
            label="training races (mean of last 100)")
    ax.plot(va.index / 1e6, va["score"], color=plot_style.INK, marker="o", markersize=6, linewidth=1.2,
            label=f"validation ({len(config.VALIDATION_DURING_TRAINING)} tracks)")
    ax.axhline(config.SOLVED_SCORE, color=plot_style.MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0.005, config.SOLVED_SCORE + 15, "solved (900)", color=plot_style.INK_SECONDARY, fontsize=9,
            transform=ax.get_yaxis_transform())
    ax.set_ylabel("Score")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"Agent {args.agent}, seed {args.seed}: score", loc="left")

    ax = axes[1]
    ax.plot(x_tr, 100 * tr["off_track"].astype(float).rolling(100, min_periods=20).mean(), color=color, linewidth=1.6)
    ax.set_ylabel("% of races")
    ax.set_title("Left the playing area (training races, last 100)", loc="left")

    ax = axes[2]
    ax.plot(tb["approx_kl"].index / 1e6, tb["approx_kl"].rolling(8, min_periods=1).mean(), color=color, linewidth=1.6)
    ax.axhspan(0.01, 0.05, color=plot_style.GRID, zorder=0)
    ax.text(0.005, 0.03, "common PPO range (0.01-0.05)", color=plot_style.INK_SECONDARY, fontsize=9,
            va="center", transform=ax.get_yaxis_transform())
    ax.set_ylabel("approx KL")
    ax.set_title("How much each training round changed the driver", loc="left")

    ax = axes[3]
    ax.plot(tb["std"].index / 1e6, tb["std"], color=color, linewidth=1.6)
    ax.set_ylabel("exploration std")
    ax.set_title("How much random 'trying things' is left in its actions", loc="left")
    ax.set_xlabel("AI decisions in training (millions)")
    ax.set_xlim(0, config.TOTAL_TIMESTEPS / 1e6)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
