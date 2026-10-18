"""
Why is the car slow? Replays recorded races (exactly, no rendering) and measures speed and gas against how sharp
the road ahead is, and how fast the car would have to go to finish a lap in time.

    python -m scripts.diagnose_speed --agent A --seed 42 --tracks validation --colors normal --out results/step2/speed_diagnosis

Upcoming turn = how much the road's direction changes over the next ~12 track points (about 40 m) from the
car's nearest point. Needed speed = lap length / 20 s (1000 frames at 50 frames per second).
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config
from racecar import plot_style
from racecar.replay import check_matches, replay

plot_style.apply()
import matplotlib.pyplot as plt

LOOKAHEAD_POINTS = 12
TURN_BINS = [0, 15, 45, 90, 360]
TURN_LABELS = ["straight (<15 deg)", "gentle (15-45)", "medium (45-90)", "sharp (>90)"]
SECONDS = 1000 / 50


def heading_change(track_xy, i, n_ahead):
    """Total absolute change of road direction (degrees) over the next n_ahead points after point i."""
    n = len(track_xy)
    pts = track_xy[[(i + k) % n for k in range(n_ahead + 2)]]
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.angle(np.exp(1j * np.diff(ang)))
    return float(np.degrees(np.abs(turn).sum()))


def analyse(args):
    row, actions = args
    rep = replay(row["track_seed"], row["color_seed"], actions, render=False)
    check_matches(rep, row)
    tel = pd.DataFrame(rep["telemetry"])
    track = rep["track_xy"]
    nearest = np.argmin(((tel[["x", "y"]].to_numpy()[:, None, :] - track[None]) ** 2).sum(-1), axis=1)
    tel["turn_ahead"] = [heading_change(track, i, LOOKAHEAD_POINTS) for i in nearest]
    tel["track_seed"] = row["track_seed"]
    lap_length = float(np.linalg.norm(np.diff(np.vstack([track, track[:1]]), axis=0), axis=1).sum())
    per_race = {
        "track_seed": row["track_seed"], "score": row["score"], "covered": row["track_covered_pct"],
        "lap": bool(row["lap_completed"]), "off_track": bool(row["off_track"]), "frames": int(row["frames"]),
        "lap_length": lap_length, "needed_speed": lap_length / SECONDS, "mean_speed": float(tel["speed"].mean()),
        "max_speed": float(tel["speed"].max()),
    }
    return per_race, tel[["track_seed", "speed", "gas", "brake", "steer", "on_road", "turn_ahead"]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="A")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipe", default=config.RECIPE)
    p.add_argument("--tracks", default="validation")
    p.add_argument("--colors", default="normal")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run = config.run_name(args.agent, args.seed, args.recipe)
    name = f"{args.tracks}_{args.colors}"
    races = pd.read_csv(config.RESULTS_DIR / "evals" / run / f"{name}.csv")
    acts = np.load(config.RUNS_DIR / run / "actions" / f"{name}.npz")
    jobs = [(r, acts[f"{r['track_seed']}_{r['color_seed']}"]) for r in races.to_dict("records")]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        results = list(pool.map(analyse, jobs))
    per_race = pd.DataFrame([r for r, _ in results])
    frames = pd.concat([t for _, t in results], ignore_index=True)
    frames["turn"] = pd.cut(frames["turn_ahead"], TURN_BINS, labels=TURN_LABELS, right=False)

    by_turn = frames.groupby("turn", observed=True).agg(
        share_of_time=("speed", lambda s: 100 * len(s) / len(frames)), mean_speed=("speed", "mean"),
        p90_speed=("speed", lambda s: s.quantile(0.9)), mean_gas=("gas", "mean"),
        pct_full_gas=("gas", lambda g: 100 * (g > 0.9).mean()), pct_braking=("brake", lambda b: 100 * (b > 0.1).mean()),
        pct_all_wheels_off_road=("on_road", lambda r: 100 * (~r).mean())).round(2)
    timeout = ~per_race["lap"] & ~per_race["off_track"] & (per_race["frames"] >= 1000)
    summary = {
        "races": len(per_race), "replays_matching": f"{len(per_race)}/{len(races)}",
        "mean_needed_speed": round(per_race["needed_speed"].mean(), 1),
        "mean_achieved_speed": round(per_race["mean_speed"].mean(), 1),
        "races_fast_enough_on_average": int((per_race["mean_speed"] >= per_race["needed_speed"]).sum()),
        "races_out_of_time": int(timeout.sum()),
        "races_out_of_time_with_90pct_covered": int((timeout & (per_race["covered"] >= 90)).sum()),
        "overall_mean_gas": round(frames["gas"].mean(), 3), "overall_pct_full_gas": round(100 * (frames["gas"] > 0.9).mean(), 2),
        "highest_gas_used": round(frames["gas"].max(), 3), "top_speed_seen": round(frames["speed"].max(), 1),
    }
    out = config.PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    per_race.round(2).to_csv(out / "per_race.csv", index=False)
    by_turn.to_csv(out / "by_upcoming_turn.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    text = (f"# Speed diagnosis: {run}, {name}\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary.items())
            + "\n\nBy how sharp the road ahead is (all frames of all races):\n\n" + by_turn.to_markdown() + "\n")
    (out / "speed_diagnosis.md").write_text(text)
    print(text)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.scatter(per_race["needed_speed"], per_race["mean_speed"], s=28, color=plot_style.AGENT_COLORS[args.agent],
               edgecolor=plot_style.SURFACE, linewidth=0.6)
    lim = [min(per_race[["needed_speed", "mean_speed"]].min()) - 3, max(per_race[["needed_speed", "mean_speed"]].max()) + 3]
    ax.plot(lim, lim, color=plot_style.MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(lim[1], lim[1], " fast enough above this line", color=plot_style.INK_SECONDARY, fontsize=9, va="bottom", ha="right")
    ax.set_xlabel("speed needed to finish the lap in time")
    ax.set_ylabel("car's average speed")
    ax.set_title("Each race (one dot)", loc="left")
    ax = axes[1]
    ax.bar(range(len(by_turn)), by_turn["mean_speed"], color=plot_style.AGENT_COLORS[args.agent], width=0.6)
    ax.set_xticks(range(len(by_turn)), [l.replace(" (", "\n(") for l in by_turn.index])
    for i, (v, g) in enumerate(zip(by_turn["mean_speed"], by_turn["mean_gas"])):
        ax.text(i, v + 0.8, f"{v:.0f}\n(gas {g:.2f})", ha="center", va="bottom", fontsize=9, color=plot_style.INK_SECONDARY)
    ax.set_ylim(0, by_turn["mean_speed"].max() * 1.25)
    ax.set_ylabel("average speed")
    ax.set_title("Speed by how sharp the road ahead is", loc="left")
    fig.tight_layout()
    fig.savefig(out / "speed_diagnosis.png", dpi=130)
    print("saved", out / "speed_diagnosis.png")


if __name__ == "__main__":
    main()
