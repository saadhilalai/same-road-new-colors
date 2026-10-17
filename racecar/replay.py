"""
Replay a recorded race exactly: same track seed, same color seed, same actions.
The simulator is deterministic, so the replay reproduces the race frame for frame.
Every replay checks its score against the recorded result.
"""

import cv2
import numpy as np

import config
from racecar.envs import make_env


def to_ai_view(state96):
    """Same shrink step the AI's ResizeObservation wrapper does (cv2 INTER_AREA to 64x64)."""
    return cv2.resize(state96, (config.OBS_SIZE, config.OBS_SIZE), interpolation=cv2.INTER_AREA)


def replay(track_seed, color_seed, actions, render=True):
    """
    Plays `actions` (one per AI decision, each held for FRAME_SKIP frames, like FrameSkip does).
    Returns a dict with per-frame telemetry, the frames (if render), and the final result.
    """
    env = make_env(render_mode="rgb_array" if render else None, ai_view=False)
    color_seed = None if color_seed is None or int(color_seed) == -1 else int(color_seed)
    state, _ = env.reset(seed=int(track_seed), options={"color_seed": color_seed})
    car = env.unwrapped
    frames, ai_views, telemetry = [], [], []
    score, done, info = 0.0, False, {}
    for decision, action in enumerate(actions):
        for _ in range(config.FRAME_SKIP):
            state, reward, term, trunc, info = env.step(np.asarray(action, dtype=np.float32))
            score += reward
            vx, vy = car.car.hull.linearVelocity
            on_road = any(len(w.tiles) > 0 for w in car.car.wheels)
            telemetry.append({
                "decision": decision,
                "x": car.car.hull.position[0],
                "y": car.car.hull.position[1],
                "speed": float(np.hypot(vx, vy)),
                "steer": float(action[0]),
                "gas": float(action[1]),
                "brake": float(action[2]),
                "on_road": on_road,
                "tiles_visited": info["tiles_visited"],
            })
            if render:
                frames.append(env.render())
                ai_views.append(state)
            if term or trunc:
                done = True
                break
        if done:
            break
    track_xy = np.array([(x, y) for _, _, x, y in car.track])
    env.close()
    return {
        "score": score,
        "frames_played": len(telemetry),
        "tiles_visited": info["tiles_visited"],
        "n_tiles": info["n_tiles"],
        "off_track": info["off_track"],
        "lap_finished_gym": info["lap_finished_gym"],
        "telemetry": telemetry,
        "track_xy": track_xy,
        "frames": frames,
        "ai_views_96": ai_views,
    }


def check_matches(replayed, recorded_row):
    """A replay is only valid if it reproduces the recorded race exactly."""
    ok = (
        abs(replayed["score"] - recorded_row["score"]) < 1e-3
        and replayed["frames_played"] == recorded_row["frames"]
        and replayed["tiles_visited"] == recorded_row["tiles_visited"]
    )
    if not ok:
        raise AssertionError(
            f"Replay mismatch on track {recorded_row['track_seed']}: replay score {replayed['score']:.3f} / "
            f"{replayed['frames_played']} frames vs recorded {recorded_row['score']:.3f} / {recorded_row['frames']} frames"
        )
    return True
