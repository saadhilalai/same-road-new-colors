"""
Building blocks for training: the parallel simulators and the PPO agent,
both configured from config.py (RL Baselines3 Zoo settings).
"""

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

import config
from racecar.envs import add_ai_view_wrappers

MONITOR_KEYS = ("tiles_visited", "n_tiles", "off_track", "lap_finished_gym", "color_seed")


def build_vec_env(n_envs, seed, color_mode, continuous, grayscale, monitor_dir=None):
    """
    n_envs simulators, each in its own process.
    Monitor sits directly on the raw env, so it logs the official score
    (before frame skip or reward rescaling touch anything).
    """
    assert n_envs <= config.MAX_SIM_PROCESSES
    venv = make_vec_env(
        config.ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(color_mode=color_mode, continuous=continuous),
        wrapper_class=add_ai_view_wrappers,
        wrapper_kwargs=dict(grayscale=grayscale),
        monitor_dir=monitor_dir,
        monitor_kwargs=dict(info_keywords=MONITOR_KEYS),
        vec_env_cls=SubprocVecEnv,
    )
    venv = VecNormalize(venv, gamma=config.PPO_PARAMS["gamma"], **config.VEC_NORMALIZE)
    return VecFrameStack(venv, n_stack=config.FRAME_STACK)


def linear_schedule(initial_value):
    """Learning rate that starts at initial_value and shrinks to 0 by the end of training."""

    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def build_ppo(venv, seed, continuous, tensorboard_log=None):
    policy_kwargs = dict(config.POLICY_KWARGS)
    policy_kwargs["activation_fn"] = getattr(torch.nn, policy_kwargs["activation_fn"])
    extra = {}
    if continuous:
        policy_kwargs.update(config.CONTINUOUS_ONLY_POLICY_KWARGS)
        extra = dict(config.CONTINUOUS_ONLY_PARAMS)
    return PPO(
        "CnnPolicy",
        venv,
        learning_rate=linear_schedule(config.LEARNING_RATE),
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=config.DEVICE,
        tensorboard_log=tensorboard_log,
        verbose=0,
        **config.PPO_PARAMS,
        **extra,
    )
