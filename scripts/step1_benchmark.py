"""
Step 1 speed test. Measures:
  A. One simulator alone (frames per second).
  B. 8 and 10 simulators in parallel, random actions (AI decisions per second).
  C. Real PPO training with the Zoo settings on the GPU, for each input/control option.
  D. Evaluation speed (10 simulators, AI choosing actions).
Uses scratch seeds only. Results go to results/step1/benchmark.json.
"""

import json
import time

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

import config
from racecar.envs import make_env
from racecar.train_utils import build_ppo, build_vec_env

OUT = config.RESULTS_DIR / "step1"
SEED = config.SCRATCH_TRACK_START
PPO_ROLLOUTS = 12          # timed training rounds per option (first one is dropped as warm-up)


class Timer(BaseCallback):
    """Records when each data-collection phase starts and ends."""

    def __init__(self):
        super().__init__()
        self.rollout_starts, self.rollout_ends = [], []

    def _on_rollout_start(self):
        self.rollout_starts.append(time.perf_counter())

    def _on_rollout_end(self):
        self.rollout_ends.append(time.perf_counter())

    def _on_step(self):
        return True


def single_env_speed():
    env = make_env(ai_view=False)
    env.reset(seed=SEED)
    env.action_space.seed(0)
    t0, frames = time.perf_counter(), 0
    while frames < 3000:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        frames += 1
        if term or trunc:
            env.reset()
    return frames / (time.perf_counter() - t0)


def parallel_env_speed(n_envs):
    venv = build_vec_env(n_envs, SEED, "normal", continuous=True, grayscale=False)
    venv.reset()
    actions = np.zeros((n_envs, 3), dtype=np.float32)
    actions[:, 1] = 0.3
    for _ in range(50):
        venv.step(actions)
    t0 = time.perf_counter()
    for _ in range(600):
        venv.step(actions)
    rate = 600 * n_envs / (time.perf_counter() - t0)
    venv.close()
    return rate


def ppo_speed(continuous, grayscale):
    venv = build_vec_env(config.N_ENVS, SEED, "normal", continuous, grayscale)
    model = build_ppo(venv, SEED, continuous)
    torch.cuda.reset_peak_memory_stats()
    timer = Timer()
    per_rollout = config.PPO_PARAMS["n_steps"] * config.N_ENVS
    model.learn(total_timesteps=PPO_ROLLOUTS * per_rollout, callback=timer)
    t_end = time.perf_counter()
    starts, ends = timer.rollout_starts, timer.rollout_ends
    # Drop round 1 (warm-up). Each round = collect data (start->end) + learn from it (end->next start).
    total = t_end - starts[1]
    collect = sum(e - s for s, e in zip(starts[1:], ends[1:]))
    steps = (PPO_ROLLOUTS - 1) * per_rollout
    venv.close()
    return {
        "decisions_per_sec": round(steps / total, 1),
        "share_of_time_collecting_data": round(collect / total, 2),
        "share_of_time_learning_on_gpu": round(1 - collect / total, 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "hours_for_4M_decisions": round(config.TOTAL_TIMESTEPS / (steps / total) / 3600, 2),
    }


def eval_speed(n_envs=10):
    """AI picks actions for 10 simulators at once (like our evaluation will)."""
    venv = build_vec_env(n_envs, SEED, "normal", continuous=True, grayscale=False)
    model = build_ppo(venv, SEED, continuous=True)
    obs = venv.reset()
    for _ in range(20):
        obs, *_ = venv.step(model.predict(obs, deterministic=True)[0])
    t0 = time.perf_counter()
    for _ in range(400):
        obs, *_ = venv.step(model.predict(obs, deterministic=True)[0])
    rate = 400 * n_envs / (time.perf_counter() - t0)
    venv.close()
    return rate


def main():
    torch.set_num_threads(config.TORCH_CPU_THREADS)
    report = {"gpu_free_gb_at_start": round(torch.cuda.mem_get_info()[0] / 2**30, 2)}
    report["A_one_simulator_frames_per_sec"] = round(single_env_speed(), 1)
    print("A", report["A_one_simulator_frames_per_sec"], flush=True)
    for n in (8, 10):
        report[f"B_{n}_simulators_decisions_per_sec"] = round(parallel_env_speed(n), 1)
        print("B", n, report[f"B_{n}_simulators_decisions_per_sec"], flush=True)
    report["C_ppo_training"] = {}
    for continuous in (True, False):
        for grayscale in (False, True):
            name = f"{'continuous' if continuous else 'discrete'}_{'gray' if grayscale else 'color'}"
            report["C_ppo_training"][name] = ppo_speed(continuous, grayscale)
            print("C", name, report["C_ppo_training"][name], flush=True)
    report["D_eval_10_simulators_decisions_per_sec"] = round(eval_speed(), 1)
    print("D", report["D_eval_10_simulators_decisions_per_sec"], flush=True)
    print(json.dumps(report, indent=2))
    (OUT / "benchmark.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
