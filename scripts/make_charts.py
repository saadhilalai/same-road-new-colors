"""
Step 7 charts (one color per agent, colorblind-safe pair from racecar.plot_style):
1. Learning curves, A vs B: training races and the 20-track checks during training.
2. Normal vs unseen colors, A vs B, on the hidden test tracks (the main result).
3. The cost of robustness: A vs B on normal colors, test tracks.

    python -m scripts.make_charts            # every chart whose data exists
Writes results/charts/*.png.
"""

import json

import numpy as np
import pandas as pd

import config
from racecar import plot_style

plot_style.apply()
import matplotlib.pyplot as plt

OUT = config.RESULTS_DIR / "charts"
NAMES = {"A": "Agent A (normal colors)", "B": "Agent B (random colors)"}
ROLLING = 100


def runs(agent):
    return [(seed, config.RUNS_DIR / config.run_name(agent, seed, config.RECIPE)) for seed in config.TRAINING_SEEDS]


def learning_curves(path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    grid = np.linspace(0, config.BETA_TOTAL_TIMESTEPS, 201)
    ends = {0: {}, 1: {}}
    for agent in ("A", "B"):
        color = plot_style.AGENT_COLORS[agent]
        train, check = [], []
        for seed, run in runs(agent):
            ep = pd.read_csv(run / "train_episodes.csv").sort_values("timesteps")
            smooth = ep["score"].rolling(ROLLING, min_periods=20).mean()
            y = np.interp(grid, ep["timesteps"], smooth.bfill())
            train.append(y)
            axes[0].plot(grid / 1e6, y, color=color, lw=1, alpha=0.3)
            v = pd.read_csv(run / "validation.csv").groupby("timesteps")["score"].mean()
            check.append(np.interp(grid, v.index, v.values))
            axes[1].plot(v.index / 1e6, v.values, color=color, lw=1, alpha=0.3)
        for i, (ax, ys) in enumerate(((axes[0], train), (axes[1], check))):
            m = np.mean(ys, axis=0)
            ax.plot(grid / 1e6, m, color=color, lw=2.6, label=NAMES[agent])
            ends[i][agent] = m[-1]
    for i, ax in enumerate(axes):   # end labels, pushed apart when the two lines end close together
        lo, hi = sorted(ends[i], key=ends[i].get)
        close = abs(ends[i][hi] - ends[i][lo]) < 40
        for agent, dy in ((lo, -8 if close else 0), (hi, 8 if close else 0)):
            ax.annotate(f"{agent} {ends[i][agent]:.0f}", (grid[-1] / 1e6, ends[i][agent]), xytext=(6, dy),
                        textcoords="offset points", va="center", fontsize=10, color=plot_style.INK_SECONDARY)
    axes[0].set_title("Training races (B's in random colors)", loc="left", fontsize=12)
    axes[1].set_title("Checks: 20 validation tracks, normal colors", loc="left", fontsize=12)
    for ax in axes:
        ax.set_xlabel("AI decisions (millions)")
        ax.set_xlim(0, config.BETA_TOTAL_TIMESTEPS / 1e6 * 1.1)
        ax.set_ylim(-130, 980)
        ax.axhline(config.SOLVED_SCORE, color=plot_style.MUTED, ls=(0, (4, 3)), lw=1.1)
        ax.text(0.08, config.SOLVED_SCORE + 14, "solved (900)", fontsize=9, color=plot_style.INK_SECONDARY)
    axes[0].text(0.08, -105, f"mean of the last {ROLLING} training races", fontsize=9, color=plot_style.INK_SECONDARY)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower right", bbox_to_anchor=(1, 0.06))
    fig.suptitle("Learning curves: thick line = mean of 3 seeds, thin lines = single seeds", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def test_scores():
    """Per-track mean over seeds, and per-seed means, for each agent and color set (test tracks)."""
    per_track, per_seed = {}, {}
    for agent in ("A", "B"):
        for colors in ("normal", "unseen"):
            frames = [pd.read_csv(config.RESULTS_DIR / "evals" / run.name / f"test_{colors}.csv")
                      for _, run in runs(agent)]
            df = pd.concat(frames)
            per_track[agent, colors] = df.groupby("track_seed")["score"].mean()
            per_seed[agent, colors] = df.groupby("seed")["score"].mean()
    return per_track, per_seed


def main_result(path, per_track, per_seed, tests):
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    x = {"normal": 0, "unseen": 1}
    for k, agent in enumerate(("A", "B")):
        color = plot_style.AGENT_COLORS[agent]
        off = (k - 0.5) * 0.08
        for seed in config.TRAINING_SEEDS:
            ys = [per_seed[agent, c][seed] for c in x]
            ax.plot([0 + off, 1 + off], ys, color=color, lw=1, alpha=0.35)
            ax.scatter([0 + off, 1 + off], ys, color=color, s=18, alpha=0.5, zorder=3, edgecolor=plot_style.SURFACE)
        means = [per_track[agent, c].mean() for c in x]
        ax.plot([0 + off, 1 + off], means, color=color, lw=3, marker="o", ms=10, label=NAMES[agent], zorder=4,
                markeredgecolor=plot_style.SURFACE, markeredgewidth=2)
        for c, m in zip(x, means):
            ax.annotate(f"{agent} {m:.0f}", (x[c] + off, m), xytext=(14 if x[c] else -14, 0), textcoords="offset points",
                        ha="left" if x[c] else "right", va="center", fontsize=11, color=plot_style.INK_SECONDARY)
    main = tests[1]
    ax.set_title(f"Unseen colors: B beats A by {main['mean_diff']:+.0f} points "
                 f"(95% CI {main['ci95'][0]:+.0f} to {main['ci95'][1]:+.0f}, Holm p = {main['p_holm']:.1g})",
                 loc="left", fontsize=11.5)
    ax.axhline(config.SOLVED_SCORE, color=plot_style.MUTED, ls=(0, (4, 3)), lw=1.1)
    ax.text(0.5, config.SOLVED_SCORE + 12, "solved (900)", fontsize=9, color=plot_style.INK_SECONDARY, ha="center")
    ax.set_xticks([0, 1], ["normal colors", "unseen colors"])
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("mean score, 100 hidden test tracks")
    ax.legend(loc="lower left")
    fig.suptitle("Final test: normal vs unseen colors (thin lines = single seeds)", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cost_of_robustness(path, per_track, per_seed, tests):
    price = tests[2]
    diff = (per_track["B", "normal"] - per_track["A", "normal"]).sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    for k, agent in enumerate(("A", "B")):
        color = plot_style.AGENT_COLORS[agent]
        m = per_track[agent, "normal"].mean()
        ax.bar(k, m, 0.6, color=color, edgecolor=plot_style.SURFACE, lw=2)
        ax.scatter([k] * 3, per_seed[agent, "normal"].values, color=plot_style.INK, s=22, zorder=3,
                   label="single seeds" if k == 0 else None)
        ax.text(k, m + 15, f"{m:.0f}", ha="center", fontsize=11, color=plot_style.INK_SECONDARY)
    ax.set_xticks([0, 1], ["Agent A", "Agent B"])
    ax.set_ylabel("mean score, normal colors")
    ax.set_title("Normal colors, 100 hidden test tracks", loc="left", fontsize=12)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center")
    ax = axes[1]
    colors = [plot_style.AGENT_COLORS["B"] if d > 0 else plot_style.AGENT_COLORS["A"] for d in diff]
    ax.bar(np.arange(len(diff)), diff.values, 1.0, color=colors, edgecolor=plot_style.SURFACE, lw=0.4)
    ax.axhline(0, color=plot_style.BASELINE, lw=1)
    ax.axhline(diff.mean(), color=plot_style.INK, lw=1.4, ls=(0, (4, 3)))
    ax.text(2, diff.mean(), f" mean {diff.mean():+.1f}", va="bottom", fontsize=10, color=plot_style.INK)
    ax.set_xticks([])
    ax.set_xlabel("tracks, sorted (blue: A better, orange: B better)")
    ax.set_ylabel("B minus A (mean of 3 seeds)")
    verdict = "significant" if price["p_holm"] < 0.05 else "not significant"
    ax.set_title(f"B minus A per track: {price['mean_diff']:+.1f} (95% CI {price['ci95'][0]:+.1f} to "
                 f"{price['ci95'][1]:+.1f}), {verdict}", loc="left", fontsize=12)
    ax.grid(axis="x", visible=False)
    fig.suptitle("The cost of robustness: does training on random colors cost points on normal colors?",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    learning_curves(OUT / "learning_curves.png")
    print("wrote learning_curves.png", flush=True)
    result = config.RESULTS_DIR / "final_test" / "color_test.json"
    if result.exists():
        tests = json.loads(result.read_text())["tests"]
        per_track, per_seed = test_scores()
        main_result(OUT / "main_result_normal_vs_unseen.png", per_track, per_seed, tests)
        cost_of_robustness(OUT / "cost_of_robustness.png", per_track, per_seed, tests)
        print("wrote the final-test charts", flush=True)


if __name__ == "__main__":
    main()
