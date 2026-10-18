"""
Step 3 color test: Agent A (normal colors) vs Agent B (random colors) on the analysis tracks,
with normal colors and with unseen colors. Applies the test plan written in PROGRESS.md before any B result existed.

    python -m scripts.compare_agents --out-dir results/step3

Unit of analysis: a track. For each of the 100 analysis tracks and each color condition, an agent's score is the
mean over its 3 seeds (final copies). Comparisons pair the same track (and, for unseen colors, the same colors).
Tests: Wilcoxon signed-rank, two-sided, on the 100 paired per-track differences; Holm correction over the 4 tests;
95% CI of the mean difference by paired bootstrap over tracks (10,000 resamples, fixed seed).
A difference is called significant only if its Holm-adjusted p < 0.05.
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
from racecar import plot_style
from racecar.evaluate import summarize

plot_style.apply()
import matplotlib.pyplot as plt

# (name, first, second): the difference is first minus second. Pre-registered in PROGRESS.md (Step 3 plan).
TESTS = [
    ("A: unseen minus normal colors (does A suffer on new colors?)", ("A", "unseen"), ("A", "normal")),
    ("Unseen colors: B minus A (main question)", ("B", "unseen"), ("A", "unseen")),
    ("Normal colors: B minus A (price of robustness if negative)", ("B", "normal"), ("A", "normal")),
    ("B: unseen minus normal colors", ("B", "unseen"), ("B", "normal")),
]
ALPHA = 0.05
BOOTSTRAP_RESAMPLES = 10_000


def load(agent, colors, track_set):
    frames = []
    for seed in config.TRAINING_SEEDS:
        path = config.RESULTS_DIR / "evals" / config.run_name(agent, seed) / f"{track_set}_{colors}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["seed"] = seed
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def bootstrap_ci(diff, rng):
    idx = rng.integers(0, len(diff), size=(BOOTSTRAP_RESAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="analysis")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    out = config.PROJECT_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    data = {(a, c): load(a, c, args.tracks) for a in config.AGENTS for c in ("normal", "unseen")}
    lines = [f"# Step 3 color test ({args.tracks} tracks, final copies, recipe '{config.RECIPE}')", ""]

    # Per seed and mean of seeds, every metric
    rows = []
    for (agent, colors), df in data.items():
        if df is None:
            continue
        for seed, g in df.groupby("seed"):
            rows.append({"agent": agent, "colors": colors, "seed": seed, **summarize(g)})
        seed_means = pd.DataFrame([r for r in rows if r["agent"] == agent and r["colors"] == colors])
        numeric = seed_means.drop(columns=["agent", "colors", "seed", "races", "solved_900"]).mean().round(1)
        rows.append({"agent": agent, "colors": colors, "seed": "mean of seeds", "races": int(seed_means["races"].sum()),
                     "solved_900": bool(numeric["mean_score"] >= config.SOLVED_SCORE), **numeric})
    table = pd.DataFrame(rows)
    lines += ["## Results per seed (and mean of seeds)", "", table.to_markdown(index=False), ""]

    # Per-track means over seeds, paired tests
    per_track = {k: df.groupby("track_seed")["score"].mean() for k, df in data.items() if df is not None}
    n_seeds = {k: df["seed"].nunique() for k, df in data.items() if df is not None}
    rng = np.random.default_rng(0)
    results = []
    for name, first, second in TESTS:
        if first not in per_track or second not in per_track:
            results.append({"test": name, "status": "not run (missing results)"})
            continue
        both = pd.concat([per_track[first], per_track[second]], axis=1, join="inner").dropna()
        diff = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
        lo, hi = bootstrap_ci(diff, rng)
        results.append({
            "test": name, "status": "run", "tracks": len(diff),
            "seeds_per_agent": f"{n_seeds[first]} vs {n_seeds[second]}",
            "mean_diff": round(float(diff.mean()), 1), "ci95": [round(float(lo), 1), round(float(hi), 1)],
            "median_diff": round(float(np.median(diff)), 1),
            "first_better": int((diff > 0).sum()), "second_better": int((diff < 0).sum()), "ties": int((diff == 0).sum()),
            "p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0,
        })
    run = [r for r in results if r["status"] == "run"]
    for r, adj in zip(run, holm(np.array([r["p"] for r in run])) if run else []):
        r["p_holm"] = float(adj)
        r["verdict"] = "significant (Holm p < 0.05)" if adj < ALPHA else "NOT significant"
    lines += ["## Paired tests (unit = track; score = mean over each agent's seeds)", ""]
    for r in results:
        if r["status"] != "run":
            lines.append(f"- {r['test']}: {r['status']}")
            continue
        lines.append(f"- **{r['test']}**: mean difference {r['mean_diff']:+.1f} (95% CI {r['ci95'][0]:+.1f} to "
                     f"{r['ci95'][1]:+.1f}), median {r['median_diff']:+.1f}; better on {r['first_better']} tracks, "
                     f"worse on {r['second_better']}, tied {r['ties']} ({r['tracks']} tracks, seeds {r['seeds_per_agent']}); "
                     f"Wilcoxon p = {r['p']:.3g}, Holm-adjusted p = {r['p_holm']:.3g} -> **{r['verdict']}**")
    lines += ["", "These tests compare the trained drivers across tracks. With 3 training runs per agent, they do not "
              "measure how much a new training run of the same recipe would vary.", ""]

    (out / "color_test.md").write_text("\n".join(lines) + "\n")
    (out / "color_test.json").write_text(json.dumps({"tests": results, "table": table.to_dict("records")}, indent=2, default=str))
    print("\n".join(lines))
    slope_chart(table, out / "normal_vs_unseen.png")


def slope_chart(table, path):
    """Mean score with normal vs unseen colors: thin lines = seeds, thick line = mean of seeds."""
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    xs = {"normal": 0, "unseen": 1}
    means = {}
    for agent, color in plot_style.AGENT_COLORS.items():
        t = table[table.agent == agent]
        if t.empty:
            continue
        for seed in config.TRAINING_SEEDS:
            s = t[t.seed == seed].set_index("colors")["mean_score"]
            if len(s) == 2:
                ax.plot([0, 1], [s["normal"], s["unseen"]], color=color, alpha=0.35, linewidth=1.2)
        m = t[t.seed == "mean of seeds"].set_index("colors")["mean_score"]
        means[agent] = m
        ax.plot([xs[c] for c in m.index], m.values, color=color, linewidth=3, marker="o", markersize=9,
                label=f"Agent {agent} ({'normal' if agent == 'A' else 'random'} training colors)")
    # Direct labels, nudged apart when the two agents are close
    for c in ("normal", "unseen"):
        vals = {a: m[c] for a, m in means.items() if c in m}
        order = sorted(vals, key=vals.get)
        close = len(vals) == 2 and abs(vals[order[0]] - vals[order[1]]) < 25
        for rank, a in enumerate(order):
            dy = (-9 if rank == 0 else 9) if close else 0
            ax.annotate(f"{a} {vals[a]:.0f}", (xs[c], vals[a]), textcoords="offset points",
                        xytext=(12 if xs[c] else -12, dy - 4), ha="left" if xs[c] else "right",
                        color=plot_style.INK_SECONDARY, fontsize=10)
    ax.axhline(config.SOLVED_SCORE, color=plot_style.MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0.5, config.SOLVED_SCORE + 10, "solved (900)", color=plot_style.INK_SECONDARY, fontsize=9, ha="center")
    ax.set_xticks([0, 1], ["normal colors", "unseen colors"])
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel("Mean score (100 analysis tracks)")
    ax.set_title("Normal vs unseen colors (thin lines: single seeds)", loc="left", fontsize=12)
    fig.legend(loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
