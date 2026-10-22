"""
Heat-map videos (Step 4) and side-by-side race videos (Step 7). Agent A and Agent B on the same track with the same
colors, next to each other, optionally with the blur-test heat map on top (bright = the AI's decision depends on
that part of the picture). Every video replays recorded races exactly and checks them against the recorded results.

    python -m scripts.attention_video --tracks analysis --track 20000 --seed 42 --colors normal unseen

Writes results/attention/videos/heatmap_<tracks>_track<T>_seed<S>_<colors>.mp4 (not in git: large; see Step 8).
"""

import argparse

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO

import config
from racecar import plot_style, video
from racecar.attention import blur_maps, patch_masks
from racecar.attention_frames import race_moments, race_pictures
from racecar.envs import one_pedal_to_game

ACTION_TOLERANCE = 1e-2     # as in scripts.attention_maps
AGENT_TITLES = {"A": "Agent A - trained on normal colors only", "B": "Agent B - trained on random colors"}


def recorded_race(agent, seed, track_set, colors, track, recipe=None, results_dir=None, checkpoint="final"):
    """The recorded result row and actions of one evaluated race."""
    run = config.run_name(agent, seed, recipe)
    suffix = "" if checkpoint == "final" else f"_ckpt{checkpoint}"
    results_dir = results_dir or config.RESULTS_DIR / "evals" / run
    df = pd.read_csv(results_dir / f"{track_set}_{colors}{suffix}.csv")
    row = df[df["track_seed"] == track].iloc[0]
    actions = np.load(config.RUNS_DIR / run / "actions" / f"{track_set}_{colors}{suffix}.npz")[f"{track}_{row.color_seed}"]
    return row, actions


def load_model(agent, seed, recipe=None, checkpoint="final"):
    model = PPO.load(config.RUNS_DIR / config.run_name(agent, seed, recipe) / "checkpoints" / f"model_{checkpoint}.zip",
                     device=config.DEVICE)
    model.policy.set_training_mode(False)
    return model


def race_heat_maps(model, row, actions, masks):
    """Blur-test heat map at every decision; checks the replay and every rebuilt picture against the record."""
    maps = []
    for m in race_moments(row.track_seed, row.color_seed, actions):
        if m[0] == "result":
            _, score, frames, tiles = m
            break
        d, stack, _, _ = m
        predicted, _ = model.predict(stack[None], deterministic=True)
        assert np.abs(one_pedal_to_game(predicted)[0] - actions[d]).max() <= ACTION_TOLERANCE, f"decision {d}"
        maps.append(blur_maps(model.policy, stack[None], masks)[0])
    check(row, score, frames, tiles)
    return np.stack(maps)


def check(row, score, frames, tiles):
    if not (abs(score - row.score) < 1e-3 and frames == row.frames and tiles == row.tiles_visited):
        raise AssertionError(f"replay of track {row.track_seed} doesn't match its record: {score:.3f} vs {row.score:.3f}")


def panel_frames(row, actions, maps=None):
    """Full-size pictures (1000x800, 50 per second) of a replayed race, with the heat map of the current decision on
    top if given; holds the last picture with "race over" once the race ended. Yields forever (the caller stops)."""
    top = np.percentile(maps, 99.5) + 1e-12 if maps is not None else None
    last = None
    for m in race_pictures(row.track_seed, row.color_seed, actions):
        if m[0] == "result":
            check(row, *m[1:])
            break
        d, picture = m
        last = video.heat_overlay(picture, maps[d], top) if maps is not None else picture
        yield last
    ended = "left the map" if row.off_track else ("lap complete" if row.lap_completed else "time up")
    last = video.label(last, f"race over ({ended}): score {row.score:.0f}", xy=(24, 24), size=30)
    while True:
        yield last


def pair_video(path, left, right, subtitle, heat=True, masks=None):
    """left/right: dict(agent, row, actions, model). One video, the two races side by side, real time."""
    streams, n = [], 0
    for side in (left, right):
        maps = race_heat_maps(side["model"], side["row"], side["actions"], masks) if heat else None
        streams.append(panel_frames(side["row"], side["actions"], maps))
        n = max(n, int(side["row"].frames))
    heads = [video.caption(1000, AGENT_TITLES[s["agent"]], subtitle, plot_style.AGENT_COLORS[s["agent"]])
             for s in (left, right)]
    with video.writer(path) as w:
        for i in range(n + video.FPS):          # one extra second on the final pictures
            panels = [np.vstack([h, next(s)]) for h, s in zip(heads, streams)]
            w.append_data(video.even(video.side_by_side(*panels)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="analysis", choices=["analysis", "validation"])
    p.add_argument("--track", type=int, default=config.ANALYSIS_TRACKS[0])
    p.add_argument("--seed", type=int, default=config.TRAINING_SEEDS[0])
    p.add_argument("--colors", nargs="+", default=["normal", "unseen"])
    args = p.parse_args()
    th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = True, False
    masks = patch_masks(device=config.DEVICE)
    for colors in args.colors:
        sides = []
        for agent in ("A", "B"):
            row, actions = recorded_race(agent, args.seed, args.tracks, colors, args.track)
            sides.append(dict(agent=agent, row=row, actions=actions, model=load_model(agent, args.seed)))
        path = config.RESULTS_DIR / "attention" / "videos" / f"heatmap_{args.tracks}_track{args.track}_seed{args.seed}_{colors}.mp4"
        pair_video(path, *sides, subtitle=f"seed {args.seed} | {args.tracks} track {args.track} | {colors} colors | "
                                          f"bright = the decision depends on it", masks=masks)
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
