"""
Step 8: data for the website. For every hidden test track: the road (the game's own polygons, so the site draws
exactly the track that was raced), both color schemes, and the path of every final-test race (A and B, 3 seeds,
normal and unseen colors), replayed from the recorded actions and checked against the recorded results.
Plus web/data/site.json with the headline numbers.

    python -m scripts.export_site_data

Writes web/data/tracks/<track>.json and web/data/site.json.
"""

import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import config
from racecar.envs import make_env, random_colors

WEB = config.PROJECT_ROOT / "web" / "data"
EVERY = 2          # one path point every 2 decisions (12.5 per second; the site interpolates)


def colors_of(color_seed):
    if color_seed == -1:
        road, bg, grass = config.NORMAL_ROAD_COLOR, config.NORMAL_BG_COLOR, config.NORMAL_GRASS_COLOR
    else:
        road, bg, grass = random_colors(color_seed)
    return {k: [int(round(c)) for c in v] for k, v in (("road", road), ("background", bg), ("grass", grass))}


def path_of(task):
    """Replays one recorded race (no drawing) and returns its path; checks it against the recorded result."""
    track, color, actions, row = task["track"], task["color"], task["actions"], task["row"]
    env = make_env(ai_view=False)
    env.reset(seed=track, options={"color_seed": None if color == -1 else color})
    car = env.unwrapped
    points, score, frames, info = [], 0.0, 0, {}
    for d, action in enumerate(actions):
        if d % EVERY == 0:
            x, y = car.car.hull.position
            points.append([round(float(x), 1), round(float(y), 1), round(float(car.car.hull.angle), 2)])
        done = False
        for _ in range(config.FRAME_SKIP):
            _, reward, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
            score += reward
            frames += 1
            if term or trunc:
                done = True
                break
        if done:
            break
    x, y = car.car.hull.position
    points.append([round(float(x), 1), round(float(y), 1), round(float(car.car.hull.angle), 2)])
    road = None
    if task.get("road"):
        road = [[[round(float(px), 1), round(float(py), 1)] for px, py in poly] + [0 if isinstance(c, np.ndarray) else
                (1 if tuple(c) == (255, 0, 0) else 2)] for poly, c in car.road_poly]
    env.close()
    ok = abs(score - row["score"]) < 1e-3 and frames == row["frames"] and info["tiles_visited"] == row["tiles_visited"]
    return {**{k: task[k] for k in ("track", "agent", "seed", "colors")}, "ok": ok, "points": points, "road": road}


def load_tasks():
    tasks = []
    for agent in ("A", "B"):
        for seed in config.TRAINING_SEEDS:
            run = config.run_name(agent, seed, config.RECIPE)
            for colors in ("normal", "unseen"):
                df = pd.read_csv(config.RESULTS_DIR / "evals" / run / f"test_{colors}.csv")
                acts = np.load(config.RUNS_DIR / run / "actions" / f"test_{colors}.npz")
                for r in df.to_dict("records"):
                    tasks.append(dict(track=int(r["track_seed"]), color=int(r["color_seed"]), agent=agent, seed=seed,
                                      colors=colors, row=r, actions=acts[f"{r['track_seed']}_{r['color_seed']}"],
                                      road=(agent, seed, colors) == ("A", 42, "normal")))
    return tasks


def main():
    tasks = load_tasks()
    rows = {(t["track"], t["agent"], t["seed"], t["colors"]): t["row"] for t in tasks}
    colors = {(t["track"], t["colors"]): t["color"] for t in tasks}
    tracks = {}
    with ProcessPoolExecutor(8, mp_context=multiprocessing.get_context("forkserver")) as pool:
        for res in pool.map(path_of, tasks, chunksize=4):
            if not res["ok"]:
                raise AssertionError(f"replay mismatch: {res['agent']} seed {res['seed']} track {res['track']} {res['colors']}")
            t = tracks.setdefault(res["track"], {"track": res["track"], "races": {}})
            if res["road"] is not None:
                t["road"] = res["road"]
            row = rows[res["track"], res["agent"], res["seed"], res["colors"]]
            t["races"].setdefault(res["colors"], {}).setdefault(res["agent"], {})[str(res["seed"])] = {
                "score": round(float(row["score"]), 1), "covered": round(float(row["track_covered_pct"]), 1),
                "lap": bool(row["lap_completed"]), "off_track": bool(row["off_track"]), "frames": int(row["frames"]),
                "path": res["points"]}
    (WEB / "tracks").mkdir(parents=True, exist_ok=True)
    for track, t in tracks.items():
        t["colors"] = {c: colors_of(colors[track, c]) for c in ("normal", "unseen")}
        t["n_tiles"] = int(rows[track, "A", 42, "normal"]["n_tiles"])
        (WEB / "tracks" / f"{track}.json").write_text(json.dumps(t, separators=(",", ":")))
    print(f"wrote {len(tracks)} tracks, {len(tasks)} races, every replay matched its record", flush=True)
    write_site_json()


def write_site_json():
    """Headline numbers for the site, read from the committed results files."""
    final = json.loads((config.RESULTS_DIR / "final_test" / "color_test.json").read_text())
    att = json.loads((config.RESULTS_DIR / "attention" / "attention_test.json").read_text())
    baseline = json.loads((config.RESULTS_DIR / "final_test" / "untrained_baseline.json").read_text())
    per_track = {}
    for agent in ("A", "B"):
        for colors in ("normal", "unseen"):
            df = pd.concat([pd.read_csv(config.RESULTS_DIR / "evals" / config.run_name(agent, s, config.RECIPE) /
                                        f"test_{colors}.csv") for s in config.TRAINING_SEEDS])
            per_track[f"{agent}_{colors}"] = df.groupby("track_seed")["score"].mean().round(1).to_dict()
    focus = {}
    for agent in ("A", "B"):
        for colors in ("normal", "unseen"):
            df = pd.concat([pd.read_csv(config.RESULTS_DIR / "attention" /
                                        f"{config.run_name(agent, s, config.RECIPE)}_analysis_{colors}.csv")
                            for s in config.TRAINING_SEEDS])
            focus[f"{agent}_{colors}"] = {r: round(float(df.groupby("track_seed")[f"focus_blur_{r}"].mean().mean()), 3)
                                          for r in ("road_edge", "road", "off_road", "car", "dashboard")}
    film = json.loads((config.RESULTS_DIR / "final_test" / "videos" / "manifest.json").read_text())
    site = {"table": final["table"], "tests": final["tests"], "attention": att, "untrained": baseline,
            "focus": focus, "film_track": film["3_a_vs_b_unseen_colors.mp4"]["track"],
            "per_track": {k: {str(t): v for t, v in d.items()} for k, d in per_track.items()},
            "tracks": config.TEST_TRACKS}
    (WEB / "site.json").write_text(json.dumps(site, indent=1, default=str))
    print("wrote site.json", flush=True)


if __name__ == "__main__":
    main()
