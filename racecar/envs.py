"""
The race-car environment and the image processing the AI sees.

Why we don't use Gymnasium's built-in `domain_randomize=True`:
    Gymnasium picks the random colors from the SAME random-number stream that
    builds the track, and it picks them first. So switching random colors on
    also changes the track you get for a given track seed. That would break
    "same track, different colors" comparisons.

Our fix (TwoSeedCarRacing):
    - The track comes from the track seed, exactly like the official env.
    - The colors come from a separate color seed, using the exact same color
      formula Gymnasium uses. Physics, rewards and time limit are unchanged.
"""

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import PLAYFIELD, CarRacing
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation

import config

cv2.setNumThreads(1)  # each simulator process uses one CPU thread


def random_colors(color_seed: int):
    """Color scheme for a color seed. Same formula as Gymnasium's domain_randomize."""
    rng = np.random.default_rng([config.COLOR_STREAM_ID, color_seed])
    road = rng.uniform(0, 210, size=3)
    bg = rng.uniform(0, 210, size=3)
    grass = np.copy(bg)
    grass[rng.integers(3)] += 20     # grass patches = background, slightly shifted in one channel
    return road, bg, grass


class TwoSeedCarRacing(CarRacing):
    """
    Official CarRacing-v3, but the track and the colors have separate seeds.

    color_mode="normal": always the default grey road / green grass.
    color_mode="random": a new random color scheme every episode (for training Agent B).
                         Its color seeds are drawn from config.TRAIN_COLOR_SEED_RANGE.
    reset(options={"color_seed": N}) forces color seed N (None = normal colors).
    """

    def __init__(self, color_mode: str = "normal", **kwargs):
        assert color_mode in ("normal", "random")
        kwargs.pop("domain_randomize", None)
        super().__init__(domain_randomize=False, **kwargs)   # built-in randomizer always off
        self.color_mode = color_mode
        self.color_seed = None
        self.track_seed = None
        self._color_seed_picker = None

    def _apply_colors(self, color_seed):
        if color_seed is None:
            self.road_color = np.array(config.NORMAL_ROAD_COLOR)
            self.bg_color = np.array(config.NORMAL_BG_COLOR)
            self.grass_color = np.array(config.NORMAL_GRASS_COLOR)
        else:
            self.road_color, self.bg_color, self.grass_color = random_colors(color_seed)

    def reset(self, *, seed=None, options=None):
        options = options or {}
        if seed is not None:
            # The training color seeds are reproducible from the env's seed.
            self._color_seed_picker = np.random.default_rng([config.COLOR_STREAM_ID, 1, seed])

        if "color_seed" in options:
            self.color_seed = options["color_seed"]
        elif self.color_mode == "random":
            if self._color_seed_picker is None:
                self._color_seed_picker = np.random.default_rng()
            self.color_seed = int(self._color_seed_picker.integers(*config.TRAIN_COLOR_SEED_RANGE))
        else:
            self.color_seed = None

        # Colors must be set BEFORE the track is built, because road tiles copy
        # the road color when they are created.
        self._apply_colors(self.color_seed)
        self.track_seed = seed   # None after the first training episode: the track then comes from the running random stream
        obs, info = super().reset(seed=seed)
        info.update(self._progress_info(off_track=False))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        x, y = self.car.hull.position
        info.update(self._progress_info(off_track=bool(abs(x) > PLAYFIELD or abs(y) > PLAYFIELD)))
        return obs, reward, terminated, truncated, info

    def _progress_info(self, off_track: bool) -> dict:
        """Numbers we report for every episode (read from the last step's info)."""
        return {
            "tiles_visited": int(self.tile_visited_count),
            "n_tiles": len(self.track),
            "off_track": off_track,
            "lap_finished_gym": bool(self.new_lap or self.tile_visited_count == len(self.track)),
            "color_seed": -1 if self.color_seed is None else int(self.color_seed),
            "track_seed": -1 if self.track_seed is None else int(self.track_seed),
        }


gym.register(
    id=config.ENV_ID,
    entry_point="racecar.envs:TwoSeedCarRacing",
    max_episode_steps=config.MAX_EPISODE_FRAMES,
    reward_threshold=config.SOLVED_SCORE,
)


class FrameSkip(gym.Wrapper):
    """Repeat each action for `skip` frames and add up the rewards (same as rl_zoo3.wrappers.FrameSkip)."""

    def __init__(self, env, skip: int):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ScheduledResets(gym.Wrapper):
    """
    For evaluation: every reset plays the next (track seed, color seed) from a list.
    When the list is used up, the simulator keeps driving a filler track whose results are ignored.
    """

    def __init__(self, env):
        super().__init__(env)
        self.schedule = []

    def set_schedule(self, episodes):
        self.schedule = list(episodes)

    def reset(self, *, seed=None, options=None):
        if self.schedule:
            track_seed, color_seed = self.schedule.pop(0)
        else:
            track_seed, color_seed = config.FILLER_TRACK, None
        return self.env.reset(seed=track_seed, options={"color_seed": color_seed})


def add_ai_view_wrappers(env, grayscale: bool, scheduled: bool = False):
    """Frame skip -> shrink to 64x64 -> (optional) grayscale. Same order as the Zoo."""
    env = FrameSkip(env, config.FRAME_SKIP)
    env = ResizeObservation(env, (config.OBS_SIZE, config.OBS_SIZE))
    if grayscale:
        env = GrayscaleObservation(env, keep_dim=True)
    if scheduled:
        env = ScheduledResets(env)
    return env


def make_env(color_mode="normal", continuous=True, grayscale=False, render_mode=None, ai_view=True):
    """One environment. ai_view=False returns the raw 96x96 env (with the time limit)."""
    env = gym.make(config.ENV_ID, color_mode=color_mode, continuous=continuous, render_mode=render_mode)
    return add_ai_view_wrappers(env, grayscale) if ai_view else env
