"""
Step 7 videos. Every clip replays a real race of the final test (hidden test tracks) exactly and is checked against
its recorded result. Clips are chosen by the rules in results/final_test/TEST_PLAN.md (fixed before the test).
Full size (1000x800 per view), 50 frames per second, H.264 at visually lossless quality.

    python -m scripts.make_videos

Writes results/final_test/videos/*.mp4 (not in git: large; published with the website) and small GIF previews in
docs/media/ (for the README).
"""

import json

import numpy as np
import pandas as pd
import torch as th
from PIL import Image

import config
from racecar import plot_style, video
from racecar.attention import blur_maps, patch_masks
from racecar.attention_frames import race_moments, race_pictures
from scripts.attention_video import AGENT_TITLES, check, load_model, pair_video, panel_frames, recorded_race

OUT = config.RESULTS_DIR / "final_test" / "videos"
MEDIA = config.PROJECT_ROOT / "docs" / "media"
SEEDS = config.TRAINING_SEEDS


def test_results(agent, colors):
    return pd.concat([pd.read_csv(config.RESULTS_DIR / "evals" / config.run_name(agent, s, config.RECIPE) /
                                  f"test_{colors}.csv") for s in SEEDS], ignore_index=True)


def race(agent, seed, colors, track, checkpoint="final", sampled=False):
    name = colors if not sampled else f"{colors}_ckpt{checkpoint}_sampled"
    run = config.run_name(agent, seed, config.RECIPE)
    df = pd.read_csv(config.RESULTS_DIR / "evals" / run / f"test_{name}.csv")
    row = df[df["track_seed"] == track].iloc[0]
    actions = np.load(config.RUNS_DIR / run / "actions" / f"test_{name}.npz")[f"{track}_{row.color_seed}"]
    return row, actions


def single_clip(path, row, actions, title, subtitle, color, start=0, length=None, header=True):
    """One race, one view (optionally a slice: start/length in frames, without a caption for B-roll)."""
    head = video.caption(1000, title, subtitle, color) if header else None
    frames = panel_frames(row, actions)
    total = int(row.frames) + (video.FPS if length is None else 0)
    with video.writer(path) as w:
        for i in range(total if length is None else start + length):
            picture = next(frames)
            if i >= start:
                w.append_data(video.even(np.vstack([head, picture]) if header else picture))


def split_screen(path, agent, seed, colors, track, masks):
    """What a human sees | what the AI sees (its real 64x64 view) | where its decision depends on the picture."""
    row, actions = race(agent, seed, colors, track)
    model = load_model(agent, seed)
    views, maps = [], []
    for m in race_moments(row.track_seed, row.color_seed, actions):
        if m[0] == "result":
            check(row, *m[1:])
            break
        _, stack, _, _ = m
        views.append(stack[..., 3:])
        maps.append(blur_maps(model.policy, stack[None], masks)[0])
    top = np.percentile(maps, 99.5) + 1e-12
    color = plot_style.AGENT_COLORS[agent]
    heads = [video.caption(1000, "What a human sees", f"{AGENT_TITLES[agent]}, seed {seed}", color),
             video.caption(800, "What the AI sees", "its real input: 64x64 pixels (+ the previous one)", color),
             video.caption(800, "What its decision depends on", "blur test: bright = matters most", color)]
    with video.writer(path) as w:
        for m in race_pictures(row.track_seed, row.color_seed, actions):
            if m[0] == "result":
                check(row, *m[1:])
                break
            d, picture = m
            ai = video.ai_view_big(views[d], 800)
            heat = video.heat_overlay(ai, maps[d], top)
            panels = [np.vstack([heads[0], picture]), np.vstack([heads[1], ai]), np.vstack([heads[2], heat])]
            w.append_data(video.even(video.side_by_side(*panels)))


def gif(video_path, path, width=720, fps=20, seconds=8):
    """A small looping preview for the README (reads only the frames it needs)."""
    import imageio.v2 as imageio
    step, frames = video.FPS // fps, []
    with imageio.get_reader(video_path) as reader:
        for i, f in enumerate(reader):
            if i >= seconds * video.FPS:
                break
            if i % step == 0:
                frames.append(Image.fromarray(f).resize((width, int(f.shape[0] * width / f.shape[1])), Image.LANCZOS))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


def main():
    th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = True, False
    masks = patch_masks(device=config.DEVICE)
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    first = config.TEST_TRACKS[0]
    manifest = {}

    def note(name, track, colors, races):
        """races: [(agent, seed, row)]: every clip's source, replay-checked (check() raises on any mismatch)."""
        manifest[name] = {"track": int(track), "colors": colors, "replays_match_recorded_results": True,
                          "races": [{"agent": a, "seed": int(sd), "score": round(float(r.score), 2), "frames": int(r.frames)}
                                    for a, sd, r in races]}

    # 1. The untrained car: first test track, normal colors, random actions from its starting policy
    row, actions = race("A", 42, "normal", first, checkpoint=0, sampled=True)
    single_clip(OUT / "1_untrained_car.mp4", row, actions, "Untrained car (before any training)",
                f"random actions | hidden test track {first} | normal colors | score {row.score:.0f}", plot_style.INK)
    note("1_untrained_car.mp4", first, "normal", [("untrained", 42, row)])

    # 2. Agent A driving well on normal colors: the A seed with the best normal-color mean, its best race
    a = test_results("A", "normal")
    seed = int(a.groupby("seed")["score"].mean().idxmax())
    best = a[a["seed"] == seed].sort_values("score").iloc[-1]
    row, actions = race("A", seed, "normal", int(best.track_seed))
    single_clip(OUT / "2_agent_a_normal_colors.mp4", row, actions, AGENT_TITLES["A"],
                f"seed {seed}, its best race of 100 | hidden test track {row.track_seed} | normal colors | "
                f"score {row.score:.0f}", plot_style.AGENT_COLORS["A"])
    note("2_agent_a_normal_colors.mp4", row.track_seed, "normal", [("A", seed, row)])

    # 3. A vs B on unseen colors, side by side, seed 42, the track with the median B-minus-A gap (a typical track)
    a42, b42 = (test_results(x, "unseen").query("seed == 42").set_index("track_seed")["score"] for x in ("A", "B"))
    gap = (b42 - a42).sort_values()
    typical = int((gap - gap.median()).abs().idxmin())
    sides = []
    for agent in ("A", "B"):
        row, actions = race(agent, 42, "unseen", typical)
        sides.append(dict(agent=agent, row=row, actions=actions, model=None))
    pair_video(OUT / "3_a_vs_b_unseen_colors.mp4", *sides, heat=False,
               subtitle=f"seed 42 | hidden test track {typical} (a typical gap) | unseen colors")
    note("3_a_vs_b_unseen_colors.mp4", typical, "unseen", [(x["agent"], 42, x["row"]) for x in sides])

    # 4. Split screen: B seed 42 on the same race
    split_screen(OUT / "4_split_screen_what_the_ai_sees.mp4", "B", 42, "unseen", typical, masks)
    note("4_split_screen_what_the_ai_sees.mp4", typical, "unseen", [("B", 42, sides[1]["row"])])

    # 5. B-roll: 10-second clean clips (from 1 s in) of 5 different tracks: B's 3 best unseen-color races,
    #    then A's and B's best normal-color races on tracks not used yet
    used, picks = set(), []
    for agent, colors, count in (("B", "unseen", 3), ("A", "normal", 1), ("B", "normal", 1)):
        for r in test_results(agent, colors).sort_values("score", ascending=False).itertuples():
            if count == 0:
                break
            if r.track_seed not in used:
                used.add(r.track_seed)
                picks.append((agent, int(r.seed), colors, int(r.track_seed)))
                count -= 1
    for i, (agent, seed, colors, track) in enumerate(picks, 1):
        row, actions = race(agent, seed, colors, track)
        name = f"5_broll_{i}_{agent}_seed{seed}_{colors}_track{track}.mp4"
        single_clip(OUT / name, row, actions, "", "", None, start=video.FPS, length=10 * video.FPS, header=False)
        note(name, track, colors, [(agent, seed, row)])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    # GIF previews for the README
    gif(OUT / "3_a_vs_b_unseen_colors.mp4", MEDIA / "a_vs_b_unseen_colors.gif", width=900, seconds=10)
    gif(OUT / "4_split_screen_what_the_ai_sees.mp4", MEDIA / "split_screen.gif", width=960, seconds=8)
    gif(OUT / "1_untrained_car.mp4", MEDIA / "untrained_car.gif", width=480, seconds=6)
    print("videos written:", *sorted(p.name for p in OUT.glob("*.mp4")), sep="\n  ")


if __name__ == "__main__":
    main()
