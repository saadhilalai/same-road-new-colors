"""
Step 4: does A look at different things than B? Applies the judging rule written in PROGRESS.md before any
A-vs-B heat-map result (2026-10-22 00:10) to the output of scripts.attention_maps.

    python -m scripts.compare_attention

Unit: a track. A track's value = the mean over an agent's 3 seeds. Four tests per method (Wilcoxon, two-sided,
Holm over the 4); a test's verdict is "supported" only if the blur test and the gradient check are both significant
in the guessed direction. Writes results/attention/attention_test.md and .json, and two figures.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import config
from racecar import plot_style
from racecar.attention_frames import REGIONS
from scripts.compare_agents import ALPHA, bootstrap_ci, holm

plot_style.apply()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = config.RESULTS_DIR / "attention"
EXAMPLES = config.RUNS_DIR / "attention_examples"
METHODS = ("blur", "gradient")
REGION_NAMES = {"road_edge": "road edge", "road": "road", "off_road": "off-road", "car": "car", "dashboard": "dashboard"}
# (name, colors, region, first agent, second agent): difference = first minus second, in the guessed direction
TESTS = [
    ("1. Normal colors: road-edge focus, B minus A", "normal", "road_edge", "B", "A"),
    ("2. Unseen colors: road-edge focus, B minus A", "unseen", "road_edge", "B", "A"),
    ("3. Normal colors: off-road focus, A minus B", "normal", "off_road", "A", "B"),
    ("4. Unseen colors: off-road focus, A minus B", "unseen", "off_road", "A", "B"),
]


def load(agent, colors):
    frames = [pd.read_csv(OUT / f"{config.run_name(agent, s, config.RECIPE)}_analysis_{colors}.csv")
              for s in config.TRAINING_SEEDS]
    return pd.concat(frames, ignore_index=True)


def per_track(df, column):
    return df.groupby("track_seed")[column].mean().dropna()   # mean over the seeds that have data


def run_tests(data):
    results = []
    rng = np.random.default_rng(0)
    for method in METHODS:
        rows = []
        for name, colors, region, first, second in TESTS:
            col = f"focus_{method}_{region}"
            both = pd.concat([per_track(data[first, colors], col), per_track(data[second, colors], col)],
                             axis=1, join="inner").dropna()
            diff = (both.iloc[:, 0] - both.iloc[:, 1]).to_numpy()
            lo, hi = bootstrap_ci(diff, rng)
            rows.append({"test": name, "method": method, "tracks": len(diff),
                         "first_mean": round(float(both.iloc[:, 0].mean()), 3),
                         "second_mean": round(float(both.iloc[:, 1].mean()), 3),
                         "mean_diff": round(float(diff.mean()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
                         "first_higher": int((diff > 0).sum()), "second_higher": int((diff < 0).sum()),
                         "p": float(wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0})
        for r, adj in zip(rows, holm(np.array([r["p"] for r in rows]))):
            r["p_holm"] = float(adj)
            r["significant"] = bool(adj < ALPHA)
            r["direction"] = "as guessed" if r["mean_diff"] > 0 else "opposite"
        results += rows
    verdicts = []
    for name, *_ in TESTS:
        b, g = [r for r in results if r["test"] == name and r["method"] == "blur"][0], \
               [r for r in results if r["test"] == name and r["method"] == "gradient"][0]
        if b["significant"] and g["significant"] and b["direction"] == g["direction"]:
            verdict = "supported" if b["direction"] == "as guessed" else "opposite"
        else:
            verdict = "not shown"
        verdicts.append({"test": name, "verdict": verdict})
    return results, verdicts


def focus_table(data, method):
    rows = []
    for (agent, colors), df in sorted(data.items()):
        row = {"agent": agent, "colors": colors, "races with moments": int((df["moments"] > 0).sum())}
        for r in REGIONS:
            row[f"{REGION_NAMES[r]} focus"] = round(per_track(df, f"focus_{method}_{r}").mean(), 2)
        for r in REGIONS:
            row[f"{REGION_NAMES[r]} share"] = f"{100 * per_track(df, f'share_{method}_{r}').mean():.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def focus_chart(data, path):
    """Focus per region (1 = the region's size), A vs B, normal and unseen colors, both methods."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), sharey="row")
    for i, method in enumerate(METHODS):
        for j, colors in enumerate(("normal", "unseen")):
            ax = axes[i, j]
            x = np.arange(len(REGIONS))
            for k, agent in enumerate(("A", "B")):
                vals = [per_track(data[agent, colors], f"focus_{method}_{r}").mean() for r in REGIONS]
                ax.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=plot_style.AGENT_COLORS[agent], label=f"Agent {agent}",
                       edgecolor=plot_style.SURFACE, linewidth=2)
            ax.axhline(1, color=plot_style.BASELINE, lw=1.2, ls="--")
            ax.set_xticks(x, [REGION_NAMES[r] for r in REGIONS])
            ax.set_title(f"{'Blur test' if method == 'blur' else 'Gradient check'}, {colors} colors", fontsize=12)
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel("focus (1 = its size's worth)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Where the AI's decisions depend on the picture (analysis tracks, 3 seeds each)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


HEAT = LinearSegmentedColormap.from_list("heat", ["#000000", "#7a1e00", "#eb6834", "#ffd166", "#ffffff"])


def overlay(view, heat, top):
    h = np.clip(heat / top, 0, 1)
    rgba = HEAT(h)[..., :3]
    a = (0.25 + 0.6 * h)[..., None]
    return (1 - a) * view / 255 + a * rgba


def example_figure(path, track="20000"):
    """The AI's view and its blur-test heat map at the same moments of one track, A vs B, normal and unseen."""
    panels = []
    for colors in ("normal", "unseen"):
        for agent in ("A", "B"):
            d = np.load(EXAMPLES / f"{config.run_name(agent, 42, config.RECIPE)}_analysis_{colors}.npz")
            if f"{track}/stacks" not in d:
                continue
            panels.append((agent, colors, d[f"{track}/decisions"], d[f"{track}/stacks"], d[f"{track}/blur"]))
    cols = 4
    fig, axes = plt.subplots(len(panels), cols, figsize=(2.3 * cols, 2.45 * len(panels)))
    for i, (agent, colors, decisions, stacks, maps) in enumerate(panels):
        top = np.percentile(maps, 99.5) + 1e-12
        for j in range(cols):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(decisions):
                ax.imshow(overlay(stacks[j][..., 3:].astype(float), maps[j], top), interpolation="nearest")
                ax.set_title(f"{decisions[j] * 2 / 50:.1f} s", fontsize=9, color=plot_style.INK_SECONDARY)
        axes[i, 0].text(-0.08, 0.5, f"Agent {agent}\n{colors}", transform=axes[i, 0].transAxes, ha="right",
                        va="center", fontsize=10, fontweight="bold", color=plot_style.AGENT_COLORS[agent])
    fig.suptitle(f"Blur-test heat maps, seed 42, analysis track {track} (bright = the decision depends on it)",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    data = {(a, c): load(a, c) for a in ("A", "B") for c in ("normal", "unseen")}
    results, verdicts = run_tests(data)
    sanity = pd.read_csv(OUT / "sanity_untrained.csv")
    trained_a = data["A", "normal"][data["A", "normal"]["seed"] == 42]

    lines = ["# Step 4: what the AI looks at (analysis tracks, final copies, recipe beta)", "",
             "Rule: PROGRESS.md, Step 4 (written before any A-vs-B result). Focus = share of the heat map in a region "
             "divided by the share of the picture the region covers (1 = its size's worth). Per track: mean of 3 seeds.", ""]
    for method in METHODS:
        lines += [f"## Focus and share per region: {'blur test (main)' if method == 'blur' else 'gradient check'}", "",
                  focus_table(data, method).to_markdown(index=False), ""]
    lines += ["## The four tests (Wilcoxon, two-sided, Holm over 4, per method)", ""]
    for r in results:
        lines.append(f"- {r['test']} [{r['method']}]: {r['first_mean']:.3f} vs {r['second_mean']:.3f}, difference "
                     f"{r['mean_diff']:+.3f} (95% CI {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f}); first higher on "
                     f"{r['first_higher']} of {r['tracks']} tracks; p = {r['p']:.3g}, Holm p = {r['p_holm']:.3g} -> "
                     f"{'significant' if r['significant'] else 'not significant'}, {r['direction']}")
    lines += ["", "## Verdicts (supported = both methods significant in the guessed direction)", ""]
    lines += [f"- {v['test']}: **{v['verdict']}**" for v in verdicts]
    agree = {k: round(float(pd.concat([data[k]["methods_agreement"]]).mean()), 2) for k in data}
    lines += ["", "## Checks", "",
              f"- How well the two methods agree (rank correlation of the two maps per picture, mean): "
              + ", ".join(f"{a} {c} {v}" for (a, c), v in agree.items()),
              f"- Sanity check, A seed 42, normal colors, same pictures: trained vs untrained network, rank correlation "
              f"of the maps: blur {sanity['trained_vs_untrained_blur'].mean():.2f}, gradient "
              f"{sanity['trained_vs_untrained_gradient'].mean():.2f} (near 1 would mean the maps only trace the picture).",
              "- Untrained network's focus (blur): " + ", ".join(
                  f"{REGION_NAMES[r]} {sanity[f'focus_blur_{r}'].mean():.2f}" for r in REGIONS)
              + "; trained A seed 42: " + ", ".join(
                  f"{REGION_NAMES[r]} {trained_a[f'focus_blur_{r}'].mean():.2f}" for r in REGIONS),
              f"- Largest gap between a rebuilt picture's action and the recorded one: "
              f"{max(df['max_action_gap'].max() for df in data.values()):.1e} (limit 1e-2).", ""]
    (OUT / "attention_test.md").write_text("\n".join(lines) + "\n")
    (OUT / "attention_test.json").write_text(json.dumps({"tests": results, "verdicts": verdicts}, indent=2))
    focus_chart(data, OUT / "focus_by_region.png")
    example_figure(OUT / "example_heat_maps.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
