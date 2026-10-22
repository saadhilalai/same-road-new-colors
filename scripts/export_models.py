"""
Step 8: the published copies of the trained models (models/), cleaned of anything machine-specific.
Stable-Baselines3 saves some training settings with each model: the folder of the training logs and the
learning-rate and clip schedules, which are saved as code and carry the path of the file that defined them. The weights
don't depend on any of that, so the published copies replace those settings with plain numbers and drop the log
folder, then check that each copy makes exactly the same decisions as the original and contains no file path.

    python -m scripts.export_models

Writes models/<run name>__final.zip for the six final agents, and models/A_seed42_beta__0.zip (untrained).
"""

import base64
import re
import zipfile

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.utils import ConstantSchedule

import config
from racecar.envs import make_env

MODELS = config.PROJECT_ROOT / "models"
# Saved by name (a Stable-Baselines3 class), not as code, so no file path comes with them
PLAIN = {"learning_rate": 0.0, "lr_schedule": ConstantSchedule(0.0),
         "clip_range": ConstantSchedule(config.BETA_PPO_PARAMS["clip_range"])}
PATH_PATTERN = re.compile(rb"/(?:home|mnt|Users)/|\\Users\\")   # Linux, WSL, macOS and Windows home folders


def sample_observations(n=64):
    """Real pictures from a scratch track (normal and random colors), stacked like the AI sees them."""
    obs = []
    for color in (None, 4242):
        env = make_env(one_pedal=True)
        o, _ = env.reset(seed=config.SCRATCH_TRACK_START + 3, options={"color_seed": color})
        prev = np.zeros_like(o)
        for i in range(n // 2):
            obs.append(np.concatenate([prev, o], -1))
            prev, (o, *_) = o, env.step(np.array([np.sin(i / 7), 0.4], np.float32))
        env.close()
    return np.stack(obs)


def has_paths(path):
    """True if any part of the saved file, including its encoded settings, mentions a file path."""
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            data = z.read(name)
            blobs = [data] + [base64.b64decode(b) for b in re.findall(rb'":serialized:": "([A-Za-z0-9+/=]+)"', data)]
            if any(PATH_PATTERN.search(b) for b in blobs):
                return True
    return False


def export(run, checkpoint, obs):
    src = config.RUNS_DIR / run / "checkpoints" / f"model_{checkpoint}.zip"
    dst = MODELS / f"{run}__{checkpoint}.zip"
    original = PPO.load(src, device="cpu")
    clean = PPO.load(src, device="cpu", custom_objects=PLAIN)
    clean.tensorboard_log = None
    clean.save(dst)
    reloaded = PPO.load(dst, device="cpu")
    same = all(th.equal(a, b) for a, b in zip(original.policy.state_dict().values(), reloaded.policy.state_dict().values()))
    a1, _ = original.predict(obs, deterministic=True)
    a2, _ = reloaded.predict(obs, deterministic=True)
    assert same and np.array_equal(a1, a2), f"{dst.name}: the cleaned copy decides differently"
    assert not has_paths(dst), f"{dst.name}: still contains a file path"
    print(f"{dst.name}: same weights, same decisions on {len(obs)} pictures, no file paths", flush=True)


def main():
    MODELS.mkdir(exist_ok=True)
    th.set_num_threads(config.TORCH_CPU_THREADS)
    obs = sample_observations()
    for agent in sorted(config.AGENTS):
        for seed in config.TRAINING_SEEDS:
            export(config.run_name(agent, seed, config.RECIPE), "final", obs)
    export(config.run_name("A", 42, config.RECIPE), "0", obs)


if __name__ == "__main__":
    main()
