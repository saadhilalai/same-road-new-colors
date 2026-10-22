"""
Step 4 (heat maps), the part without the AI: replay a recorded race and return, at chosen decisions, exactly the
images the AI saw (its 64x64 view, the last 2 stacked) and what each of its pixels shows: the road's edge, the
road, off the road, the car or the dashboard. Free of PyTorch, so many replays can run in parallel cheaply.

Which pixels are which comes from drawing the same moment again in flat colors (TwoSeedCarRacing.render_labels),
so it works the same in every color scheme.
"""

import cv2
import numpy as np

import config
from racecar.envs import make_env
from racecar.replay import to_ai_view

REGIONS = ["road_edge", "road", "off_road", "car", "dashboard"]
LABEL_ROAD, LABEL_KERB, LABEL_OFF_ROAD = (255, 0, 0), (0, 255, 0), (0, 0, 255)
EDGE_PX_96 = 3          # road edge = within 3 pixels of the road's border, either side, in the 96x96 image
                        # (2 pixels in the AI's 64x64 view), plus the kerbs
DASHBOARD_ROWS_96 = 12  # the black bar at the bottom: 5/40 of the image height (Gymnasium's _render_indicators)
ZOOM_DECISIONS = 25     # during the first second (50 frames = 25 decisions) the camera zooms in: not sampled
SAMPLE_EVERY = 5        # then every 5th decision


def region_masks_96(car_env):
    """(5, 96, 96) bool, one region per pixel of the 96x96 image (the dashboard wins, then the car)."""
    labels = car_env.render_labels(LABEL_ROAD, LABEL_KERB, LABEL_OFF_ROAD, draw_car=False).astype(int)
    with_car = car_env.render_labels(LABEL_ROAD, LABEL_KERB, LABEL_OFF_ROAD, draw_car=True).astype(int)
    kind = labels.argmax(axis=-1)                       # 0 road, 1 kerb, 2 off road
    kind[labels.max(axis=-1) < 64] = 2                  # outside the map (black) counts as off the road
    road = kind <= 1
    road[-DASHBOARD_ROWS_96:] = road[-DASHBOARD_ROWS_96 - 1]   # the road goes on under the dashboard: no false edge there
    inside =cv2.distanceTransform(road.astype(np.uint8), cv2.DIST_L2, 3)      # road pixel -> nearest non-road
    outside = cv2.distanceTransform((~road).astype(np.uint8), cv2.DIST_L2, 3)  # off-road pixel -> nearest road
    edge = (road & (inside <= EDGE_PX_96)) | (~road & (outside <= EDGE_PX_96)) | (kind == 1)
    dashboard = np.zeros_like(road)
    dashboard[-DASHBOARD_ROWS_96:] = True
    car = np.any(with_car != labels, axis=-1) & ~dashboard
    rest = ~dashboard & ~car
    return np.stack([edge & rest, road & ~edge & rest, ~road & ~edge & rest, car, dashboard])


def region_fractions(car_env):
    """(5, 64, 64) float32: for each pixel of the AI's 64x64 view, the share of it showing each region."""
    masks = region_masks_96(car_env).astype(np.float32)
    return np.stack([cv2.resize(m, (config.OBS_SIZE, config.OBS_SIZE), interpolation=cv2.INTER_AREA) for m in masks])


def sampled_decisions(n_decisions):
    return list(range(ZOOM_DECISIONS, n_decisions, SAMPLE_EVERY))


def race_moments(track_seed, color_seed, actions, full_frames=False):
    """
    Replays a recorded race and yields, before each decision d: (d, the AI's stacked view (64, 64, 6) uint8, the
    simulator, the full-size 1000x800 picture or None). At the end, yields ("result", score, frames, tiles visited).
    The AI chose action d from the images after decisions d-2 and d-1 (the start image counts as "after -1",
    and before it the stack holds zeros, like Stable-Baselines3's VecFrameStack).
    """
    env = make_env(ai_view=False, render_mode="rgb_array" if full_frames else None)
    color = None if int(color_seed) == -1 else int(color_seed)
    state, _ = env.reset(seed=int(track_seed), options={"color_seed": color})
    car = env.unwrapped
    older, latest = np.zeros((config.OBS_SIZE, config.OBS_SIZE, 3), np.uint8), to_ai_view(state)
    score, frames, info, done = 0.0, 0, {}, False
    for d, action in enumerate(np.asarray(actions, dtype=np.float32)):
        picture = None
        if full_frames:
            env.render()
            picture = np.transpose(_surface_pixels(car.surf), (1, 0, 2))   # 1000x800, as drawn
        yield d, np.concatenate([older, latest], axis=-1), car, picture
        for _ in range(config.FRAME_SKIP):
            state, reward, term, trunc, info = env.step(action)
            score += reward
            frames += 1
            if term or trunc:
                done = True
                break
        older, latest = latest, to_ai_view(state)
        if done:
            break
    env.close()
    yield "result", score, frames, info.get("tiles_visited")


def replay_for_attention(task):
    """
    task: dict(track_seed, color_seed (-1 = normal), actions (n, 3) game actions).
    Returns, at the sampled decisions, the AI's stacked view (n, 64, 64, 6) uint8 and region shares
    (n, 5, 64, 64) float16, plus the race's result (to check against the recorded one).
    """
    actions = np.asarray(task["actions"], dtype=np.float32)
    want = set(sampled_decisions(len(actions)))
    stacks, fractions, taken = [], [], []
    for moment in race_moments(task["track_seed"], task["color_seed"], actions):
        if moment[0] == "result":
            _, score, frames, tiles = moment
            break
        d, stack, car, _ = moment
        if d in want:
            stacks.append(stack)
            fractions.append(region_fractions(car).astype(np.float16))
            taken.append(d)
    return {
        "track_seed": int(task["track_seed"]), "color_seed": int(task["color_seed"]),
        "decisions": np.array(taken, dtype=np.int32),
        "stacks": np.stack(stacks) if stacks else np.zeros((0, 64, 64, 6), np.uint8),
        "fractions": np.stack(fractions) if fractions else np.zeros((0, 5, 64, 64), np.float16),
        "actions": actions[taken] if taken else np.zeros((0, 3), np.float32),
        "score": score, "frames": frames, "tiles_visited": tiles,
    }


def _surface_pixels(surf):
    import pygame
    return pygame.surfarray.array3d(surf)
