"""
Learning curves: score vs. AI decisions made in training.

    python -m scripts.plot_learning_curves --agents A --out results/step2/learning_curves.png

Left: training races (smoothed, mean of the last 100 races).
Right: validation checks (20 validation tracks, normal colors) at every checkpoint.
One color per agent; seeds as line styles; the thick line is the mean over seeds.
"""

import argparse

import numpy as np
import pandas as pd

import config
from racecar import plot_style

plot_style.apply()
import matplotlib.pyplot as plt

SEED_STYLES = {42: "-", 142: "--", 242: ":"}


def load(agent, seed, name):
    path = config.RUNS_DIR / f"{agent}_seed{seed}" / name
    return pd.read_csv(path) if path.exists() else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=["A"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for agent in args.agents:
        color = plot_style.AGENT_COLORS[agent]
        val_curves = []
        for seed in config.TRAINING_SEEDS:
            tr = load(agent, seed, "train_episodes.csv")
            va = load(agent, seed, "validation.csv")
            if tr is None or va is None:
                continue
            tr = tr.sort_values("timesteps")
            smooth = tr["score"].rolling(100, min_periods=20).mean()
            ax_train.plot(tr["timesteps"] / 1e6, smooth, color=color, linestyle=SEED_STYLES[seed],
                          linewidth=1.3, alpha=0.9, label=f"{agent} seed {seed}")
            v = va.groupby("timesteps")["score"].mean()
            val_curves.append(v)
            ax_val.plot(v.index / 1e6, v.values, color=color, linestyle=SEED_STYLES[seed], linewidth=1.3,
                        marker="o", markersize=4, alpha=0.9, label=f"{agent} seed {seed}")
        if len(val_curves) >= 2:
            common = sorted(set.intersection(*[set(v.index) for v in val_curves]))
            mean = np.mean([v.loc[common].values for v in val_curves], axis=0)
            ax_val.plot(np.array(common) / 1e6, mean, color=color, linewidth=3.2, label=f"{agent} mean of seeds")

    for ax, title in ((ax_train, "Training races (mean of last 100)"),
                      (ax_val, f"Validation: {len(config.VALIDATION_DURING_TRAINING)} tracks, normal colors")):
        ax.axhline(config.SOLVED_SCORE, color=plot_style.MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.text(0.01, config.SOLVED_SCORE + 12, "solved (900)", color=plot_style.INK_SECONDARY, fontsize=9,
                transform=ax.get_yaxis_transform())
        ax.set_title(title, loc="left")
        ax.set_xlabel("AI decisions in training (millions)")
        ax.set_xlim(0, config.TOTAL_TIMESTEPS / 1e6)
    ax_train.set_ylabel("Score")
    ax_val.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
