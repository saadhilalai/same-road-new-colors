"""
Evaluation: run a trained AI on an exact list of (track seed, color seed) races,
several simulators in parallel, and return one result row per race.

The AI always picks its single best action (deterministic=True): no random exploration.
"""

import numpy as np
import pandas as pd
import torch

import config
from racecar.train_utils import build_eval_vec_env

NORMAL = -1   # color_seed value meaning "normal colors"


def episode_record(info):
    """The project's metrics for one finished race (definitions in config.py)."""
    tiles, n_tiles = info["tiles_visited"], info["n_tiles"]
    return {
        "track_seed": info["track_seed"],
        "color_seed": info["color_seed"],
        "score": float(info["episode"]["r"]),
        "frames": int(info["episode"]["l"]),
        "tiles_visited": tiles,
        "n_tiles": n_tiles,
        "track_covered_pct": 100.0 * tiles / n_tiles,
        "lap_completed": tiles == n_tiles,
        "lap_finished_gym": bool(info["lap_finished_gym"]),
        "off_track": bool(info["off_track"]),
    }


def evaluate(model, episodes, venv=None, continuous=True, grayscale=False, record_actions=False):
    """
    episodes: list of (track_seed, color_seed); color_seed None = normal colors.
    venv: an evaluation env from build_eval_vec_env (reused during training); built here if None.
    Returns a DataFrame (one row per race) and, if record_actions, {(track, color): actions array}.
    """
    episodes = [(int(t), None if c is None else int(c)) for t, c in episodes]
    assert len(set(episodes)) == len(episodes), "duplicate races"
    assert all(t != config.FILLER_TRACK for t, _ in episodes)

    own_venv = venv is None
    if own_venv:
        venv = build_eval_vec_env(min(config.EVAL_N_ENVS, len(episodes)), continuous, grayscale)
    n = venv.num_envs
    for i in range(n):
        venv.env_method("set_schedule", episodes[i::n], indices=[i])
    pending = {(t, NORMAL if c is None else c) for t, c in episodes}

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rows, actions, current = [], {}, [[] for _ in range(n)]
    obs = venv.reset()
    while pending:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = venv.step(act)
        for i in range(n):
            if record_actions:
                current[i].append(np.array(act[i], dtype=np.float32))
            if dones[i]:
                key = (infos[i]["track_seed"], infos[i]["color_seed"])
                if key in pending:            # filler races are ignored
                    pending.remove(key)
                    rows.append(episode_record(infos[i]))
                    if record_actions:
                        actions[key] = np.stack(current[i])
                current[i] = []
    if own_venv:
        venv.close()
    df = pd.DataFrame(rows).sort_values(["track_seed", "color_seed"]).reset_index(drop=True)
    return (df, actions) if record_actions else df


def summarize(df):
    """One line of headline numbers for a set of races."""
    return {
        "races": len(df),
        "mean_score": round(df["score"].mean(), 1),
        "std_score": round(df["score"].std(ddof=1), 1) if len(df) > 1 else 0.0,
        "solved_900": bool(df["score"].mean() >= config.SOLVED_SCORE),
        "mean_track_covered_pct": round(df["track_covered_pct"].mean(), 1),
        "lap_completed_pct": round(100 * df["lap_completed"].mean(), 1),
        "lap_finished_gym_pct": round(100 * df["lap_finished_gym"].mean(), 1),
        "off_track_pct": round(100 * df["off_track"].mean(), 1),
    }
