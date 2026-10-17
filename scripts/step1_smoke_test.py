"""
Step 1 smoke test. Checks, and saves pictures for:
  1. A random (untrained) car drives one episode; rendering works (video + stills).
  2. Our env with normal colors is pixel-for-pixel identical to official CarRacing-v3.
  3. Color randomization visibly changes the colors, while the track stays the same.
  4. Gymnasium's built-in domain_randomize changes the track too (the reason for our fix).
  5. Same seed + same actions = same result (determinism).
  6. What the AI sees: 96x96 image -> 64x64 -> grayscale option -> 2 stacked frames.
Only scratch track seeds (90000+) are used here. Validation/analysis/test tracks are untouched.
"""

import json

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_checker import check_env

import config
from racecar.envs import make_env, random_colors

OUT = config.RESULTS_DIR / "step1"
TRACK = config.SCRATCH_TRACK_START
NOOP = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def run_actions(env, seed, actions, options=None):
    obs, _ = env.reset(seed=seed, options=options)
    observations, rewards = [obs], []
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        observations.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return np.array(observations), np.array(rewards)


def random_car_episode():
    """1. An untrained car pressing random buttons for one full episode."""
    env = make_env(render_mode="rgb_array", ai_view=False)
    env.action_space.seed(0)
    env.reset(seed=TRACK)
    frames, score, steps = [env.render()], 0.0, 0
    while True:
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        frames.append(env.render())
        score += r
        steps += 1
        if term or trunc:
            break
    imageio.mimsave(OUT / "random_car.mp4", frames, fps=50, quality=8, macro_block_size=1)
    imageio.imwrite(OUT / "random_car_human_view.png", frames[300])
    env.close()
    return {
        "score": round(score, 1),
        "frames": steps,
        "tiles_visited": info["tiles_visited"],
        "n_tiles": info["n_tiles"],
        "track_covered_pct": round(100 * info["tiles_visited"] / info["n_tiles"], 1),
        "off_track": info["off_track"],
        "ended_by": "time limit" if trunc else "terminated",
    }


def matches_official_env():
    """2. Same seed + same random actions in both envs -> identical pixels and rewards."""
    rng = np.random.default_rng(1)
    actions = rng.uniform([-1, 0, 0], [1, 1, 1], size=(300, 3)).astype(np.float32)
    ours = make_env(ai_view=False)
    official = gym.make(config.BASE_ENV_ID)
    o1, r1 = run_actions(ours, TRACK, actions)
    o2, r2 = run_actions(official, TRACK, actions)
    same_track = np.array_equal(ours.unwrapped.track, official.unwrapped.track)
    return bool(same_track and np.array_equal(o1, o2) and np.array_equal(r1, r2))


def color_checks():
    """3 and 4. Our color seeds keep the track; the built-in randomizer changes it."""
    env = make_env(render_mode="rgb_array", ai_view=False)
    color_seeds = [None, 90000, 90001, 90002, 90003, 90004]
    human, ai, tracks = [], [], []
    for cs in color_seeds:
        env.reset(seed=TRACK, options={"color_seed": cs})
        for _ in range(60):  # wait for the zoom-in animation to finish
            obs, *_ = env.step(NOOP)
        human.append(env.render())
        ai.append(obs)
        tracks.append(np.array(env.unwrapped.track))
    track_unchanged = all(np.array_equal(tracks[0], t) for t in tracks[1:])
    pixel_change = [float(np.abs(ai[0].astype(int) - a.astype(int)).mean()) for a in ai[1:]]

    builtin_normal = gym.make(config.BASE_ENV_ID, domain_randomize=False)
    builtin_random = gym.make(config.BASE_ENV_ID, domain_randomize=True, render_mode="rgb_array")
    builtin_normal.reset(seed=TRACK)
    builtin_frames = []
    for k in range(3):
        builtin_random.reset(seed=TRACK + k)
        for _ in range(60):
            builtin_random.step(NOOP)
        builtin_frames.append(builtin_random.render())
    builtin_random.reset(seed=TRACK)
    builtin_changes_track = not np.array_equal(
        np.array(builtin_normal.unwrapped.track), np.array(builtin_random.unwrapped.track)
    )

    labels = ["normal colors"] + [f"color seed {c}" for c in color_seeds[1:]]
    fig, axes = plt.subplots(2, 6, figsize=(18, 5.2))
    for i in range(6):
        axes[0, i].imshow(human[i])
        axes[0, i].set_title(labels[i], fontsize=10)
        axes[1, i].imshow(ai[i], interpolation="nearest")
        axes[1, i].set_title("AI's 96x96 view", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Same track (seed {TRACK}), different color seeds", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "color_randomization.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for ax, f in zip(axes, builtin_frames):
        ax.imshow(f)
        ax.axis("off")
    fig.suptitle(f"Gymnasium's built-in domain_randomize=True, seeds {TRACK}-{TRACK + 2}", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "builtin_domain_randomize.png", dpi=110)
    plt.close(fig)

    colors = {str(c): [np.round(x, 1).tolist() for x in random_colors(c)] for c in color_seeds[1:]}
    return {
        "track_unchanged_across_color_seeds": bool(track_unchanged),
        "mean_pixel_change_vs_normal_0_to_255": [round(p, 1) for p in pixel_change],
        "color_values_road_bg_grass": colors,
        "builtin_domain_randomize_changes_the_track": bool(builtin_changes_track),
    }


def determinism():
    """5. Run the same seed and actions twice."""
    rng = np.random.default_rng(2)
    actions = rng.uniform([-1, 0, 0], [1, 1, 1], size=(300, 3)).astype(np.float32)
    env = make_env(color_mode="random", ai_view=False)
    o1, r1 = run_actions(env, TRACK + 1, actions, options={"color_seed": 90010})
    o2, r2 = run_actions(env, TRACK + 1, actions, options={"color_seed": 90010})
    return bool(np.array_equal(o1, o2) and np.array_equal(r1, r2))


def what_the_ai_sees():
    """6. The processing steps, as one picture."""
    env_raw = make_env(render_mode="rgb_array", ai_view=False)
    env_rgb = make_env(grayscale=False)
    env_gray = make_env(grayscale=True)
    steer_gas = np.array([0.0, 0.5, 0.0], dtype=np.float32)
    for e in (env_raw, env_rgb, env_gray):
        e.reset(seed=TRACK + 2)
    for _ in range(80):
        raw, *_ = env_raw.step(steer_gas)
    gray_frames = []
    for _ in range(40):  # 40 decisions x frame skip 2 = the same 80 frames
        rgb, *_ = env_rgb.step(steer_gas)
        g, *_ = env_gray.step(steer_gas)
        gray_frames.append(g[..., 0])
    human = env_raw.render()

    fig, axes = plt.subplots(1, 6, figsize=(21, 3.9))
    panels = [
        (human, "What a human sees (600x400)"),
        (raw, "What the env gives the AI (96x96)"),
        (rgb, "Shrunk to 64x64 (color)"),
        (gray_frames[-1], "64x64 grayscale option"),
        (gray_frames[-2], "Stack: previous frame"),
        (gray_frames[-1], "Stack: current frame"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "what_the_ai_sees.png", dpi=110)
    plt.close(fig)
    return {
        "raw_obs_shape": list(raw.shape),
        "processed_color_shape": list(rgb.shape),
        "processed_gray_shape": list(env_gray.observation_space.shape),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    check_env(make_env(grayscale=False), warn=False)
    report = {
        "1_random_car": random_car_episode(),
        "2_identical_to_official_env_with_normal_colors": matches_official_env(),
        "3_4_colors": color_checks(),
        "5_deterministic": determinism(),
        "6_ai_view": what_the_ai_sees(),
    }
    print(json.dumps(report, indent=2))
    (OUT / "smoke_test.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
