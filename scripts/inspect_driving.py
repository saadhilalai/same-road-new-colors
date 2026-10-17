"""
How does the AI actually drive? Replays recorded races (exactly) and measures driving style.

    python -m scripts.inspect_driving --agent A --seed 42 --tracks analysis --colors normal

For every race: speed, time on grass, steering smoothness, wobble, braking, direction, possible shortcuts.
For 5 races (best, median, worst score + 2 more): MP4 video, contact sheet (frames every 0.4 s),
and a top-down map of the car's path coloured by speed. Every replay is checked against its recorded result.
Outputs go to results/driving/<agent>_seed<seed>_<tracks>_<colors>/.
"""

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import imageio.v2 as imageio
import numpy as np
import pandas as pd

import config
from racecar import plot_style
from racecar.replay import check_matches, replay

plot_style.apply()
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

SECONDS_PER_FRAME = 1 / 50


def style_stats(row, actions, rep):
    """Driving-style numbers for one race."""
    tel = pd.DataFrame(rep["telemetry"])
    steer = actions[:, 0]
    n_decisions = len(actions)
    big_steer = steer[np.abs(steer) > 0.1]
    flips = int(np.sum(np.diff(np.sign(big_steer)) != 0)) if len(big_steer) > 1 else 0
    seconds = len(tel) * SECONDS_PER_FRAME

    # Progress along the track: nearest centre-line point each frame. Big jumps = crossing grass to
    # another part of the track (a possible shortcut). Mostly decreasing index = driving backwards.
    track = rep["track_xy"]
    pos = tel[["x", "y"]].to_numpy()
    nearest = np.argmin(((pos[:, None, :] - track[None, :, :]) ** 2).sum(-1), axis=1)
    n = len(track)
    step = (np.diff(nearest) + n // 2) % n - n // 2          # wrap-around safe change in track index
    return {
        "track_seed": row["track_seed"],
        "score": row["score"],
        "track_covered_pct": row["track_covered_pct"],
        "lap_completed": row["lap_completed"],
        "seconds": round(seconds, 2),
        "mean_speed": round(tel["speed"].mean(), 2),
        "pct_time_all_wheels_on_grass": round(100 * (~tel["on_road"]).mean(), 1),
        "mean_abs_steer_change": round(float(np.mean(np.abs(np.diff(steer)))) if n_decisions > 1 else 0.0, 3),
        "steer_direction_flips_per_sec": round(flips / seconds, 2) if seconds else 0.0,
        "pct_decisions_braking": round(100 * float(np.mean(actions[:, 2] > 0.1)), 1),
        "pct_decisions_full_gas": round(100 * float(np.mean(actions[:, 1] > 0.9)), 1),
        "pct_decisions_gas_and_brake": round(100 * float(np.mean((actions[:, 1] > 0.1) & (actions[:, 2] > 0.1))), 1),
        "pct_frames_moving_backwards_along_track": round(100 * float(np.mean(step < 0)), 1),
        "possible_shortcuts": int(np.sum(np.abs(step) > 10)),
    }


def analyse_one(args):
    row, actions = args
    rep = replay(row["track_seed"], row["color_seed"], actions, render=False)
    check_matches(rep, row)
    return style_stats(row, actions, rep)


def contact_sheet(frames, title, path, every=20, cols=8):
    picks = list(range(0, len(frames), every))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 1.75))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), picks):
        ax.imshow(frames[i])
        ax.set_title(f"{i * SECONDS_PER_FRAME:.1f} s", fontsize=8, color=plot_style.INK_SECONDARY)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def path_map(rep, title, path):
    tel = pd.DataFrame(rep["telemetry"])
    track = np.vstack([rep["track_xy"], rep["track_xy"][:1]])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(track[:, 0], track[:, 1], color=plot_style.GRID, linewidth=14, solid_capstyle="round", zorder=1)
    ax.plot(track[:, 0], track[:, 1], color=plot_style.BASELINE, linewidth=1, linestyle="--", zorder=2)
    pts = tel[["x", "y"]].to_numpy()
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list("speed", plot_style.SEQUENTIAL_BLUES)
    lc = LineCollection(segs, cmap=cmap, linewidth=2.2, zorder=3)
    lc.set_array(tel["speed"].to_numpy()[:-1])
    ax.add_collection(lc)
    ax.scatter(*pts[0], s=70, color=plot_style.INK, zorder=4, label="start")
    ax.scatter(*pts[-1], s=70, facecolor=plot_style.SURFACE, edgecolor=plot_style.INK, linewidth=2, zorder=4, label="end")
    fig.colorbar(lc, ax=ax, shrink=0.6, label="speed")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--tracks", default="analysis")
    p.add_argument("--colors", default="normal")
    p.add_argument("--checkpoint", default="final")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    name = f"{args.tracks}_{args.colors}" + ("" if args.checkpoint == "final" else f"_ckpt{args.checkpoint}")
    races = pd.read_csv(config.RESULTS_DIR / "evals" / f"{args.agent}_seed{args.seed}" / f"{name}.csv")
    acts = np.load(config.RUNS_DIR / f"{args.agent}_seed{args.seed}" / "actions" / f"{name}.npz")
    out = config.RESULTS_DIR / "driving" / f"{args.agent}_seed{args.seed}_{name}"
    (out / "videos").mkdir(parents=True, exist_ok=True)

    jobs = [(r, acts[f"{r['track_seed']}_{r['color_seed']}"]) for r in races.to_dict("records")]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        stats = pd.DataFrame(list(pool.map(analyse_one, jobs)))
    stats.to_csv(out / "driving_stats.csv", index=False)
    summary = stats.drop(columns=["track_seed"]).mean(numeric_only=True).round(2).to_dict()
    summary["replays_matching_recorded_results"] = f"{len(stats)}/{len(races)}"
    (out / "driving_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    ranked = races.sort_values("score").reset_index(drop=True)
    picks = {"worst": ranked.iloc[0], "median": ranked.iloc[len(ranked) // 2], "best": ranked.iloc[-1]}
    others = [r for _, r in races.iterrows() if r["track_seed"] not in {x["track_seed"] for x in picks.values()}]
    picks.update({"example1": others[0], "example2": others[1]})
    for label, row in picks.items():
        rep = replay(row["track_seed"], row["color_seed"], acts[f"{row['track_seed']}_{row['color_seed']}"], render=True)
        check_matches(rep, row)
        tag = f"{label}_track{row['track_seed']}_score{row['score']:.0f}"
        imageio.mimsave(out / "videos" / f"{tag}.mp4", rep["frames"], fps=50, macro_block_size=1,
                        ffmpeg_params=["-crf", "18", "-preset", "slow"])
        title = (f"{args.agent} seed {args.seed}, {label}: track {row['track_seed']}, score {row['score']:.1f}, "
                 f"covered {row['track_covered_pct']:.1f}%")
        contact_sheet(rep["frames"], title, out / f"{tag}_frames.png")
        path_map(rep, title, out / f"{tag}_path.png")
        print(f"rendered {tag} (replay matches recorded result)", flush=True)


if __name__ == "__main__":
    main()
