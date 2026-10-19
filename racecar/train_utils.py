"""
Building blocks for training: the parallel simulators and the PPO agent,
both configured from config.py (RL Baselines3 Zoo settings).
"""

import multiprocessing

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize

import config
from racecar.beta_policy import BetaCnnPolicy, NormalViewCNN, SlaouiPPO
from racecar.envs import add_ai_view_wrappers

MONITOR_KEYS = ("tiles_visited", "n_tiles", "off_track", "lap_finished_gym", "color_seed", "track_seed")

# Memory fix (Step 2): each simulator process starts as a copy of a small "forkserver" process.
# Loading the big libraries there ONCE lets all simulators share them, instead of each
# re-loading its own private copy of PyTorch and pandas (measured: ~330 MB per simulator).
multiprocessing.set_forkserver_preload(["racecar.envs", "stable_baselines3", "pandas"])


def _sim_processes(n_envs, seed, color_mode, continuous, grayscale, scheduled, one_pedal=False, copies=None):
    """
    n_envs simulators, each in its own process.
    Monitor sits directly on the raw env, so it logs the official score
    (before one pedal, frame skip or reward rescaling touch anything).
    """
    assert n_envs <= config.MAX_SIM_PROCESSES
    return make_vec_env(
        config.ENV_ID,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=dict(color_mode=color_mode, continuous=continuous),
        wrapper_class=add_ai_view_wrappers,
        wrapper_kwargs=dict(grayscale=grayscale, scheduled=scheduled, resumable=not scheduled,
                            one_pedal=one_pedal, copies=copies),
        monitor_kwargs=dict(info_keywords=MONITOR_KEYS),
        vec_env_cls=SubprocVecEnv,
    )


def build_vec_env(n_envs, seed, color_mode, continuous, grayscale, vecnormalize_path=None, one_pedal=False, copies=False):
    """Training simulators: reward rescaling (Zoo) + stack of the last 2 images. copies=True: Slaoui's recolored copies."""
    venv = _sim_processes(n_envs, seed, color_mode, continuous, grayscale, scheduled=False, one_pedal=one_pedal,
                          copies="recolor" if copies else None)
    if vecnormalize_path is None:
        venv = VecNormalize(venv, gamma=config.PPO_PARAMS["gamma"], **config.VEC_NORMALIZE)
    else:
        venv = VecNormalize.load(str(vecnormalize_path), venv)   # resume: keep the reward-scaling statistics
    return VecFrameStack(venv, n_stack=config.FRAME_STACK)


def build_eval_vec_env(n_envs, continuous, grayscale, one_pedal=False, copies=False):
    """
    Evaluation simulators. No reward rescaling: it only affects learning, never what the AI sees
    (norm_obs is off), so the AI gets exactly the same images as in training.
    copies=True (a Slaoui agent): the copy channels just repeat the image; the AI never looks at them.
    """
    venv = _sim_processes(n_envs, None, "normal", continuous, grayscale, scheduled=True, one_pedal=one_pedal,
                          copies="duplicate" if copies else None)
    return VecFrameStack(venv, n_stack=config.FRAME_STACK)


def linear_schedule(initial_value):
    """Learning rate that starts at initial_value and shrinks to 0 by the end of training."""

    def schedule(progress_remaining):
        return progress_remaining * initial_value

    return schedule


def view_of(model):
    """Which simulator wrappers a trained model needs (config.recipe_view), read from the model itself."""
    channels = model.observation_space.shape[0]   # channels first (SB3 transposes images)
    return dict(one_pedal=model.action_space.shape == (2,), copies=channels == 2 * 3 * config.FRAME_STACK)


def algorithm_class(recipe):
    return SlaouiPPO if config.recipe_view(recipe)["copies"] else PPO


def build_ppo(venv, seed, continuous, tensorboard_log=None, recipe="zoo"):
    if recipe in config.BETA_RECIPES:
        return build_beta_ppo(venv, seed, tensorboard_log, recipe)
    policy_kwargs = dict(config.POLICY_KWARGS)
    policy_kwargs["activation_fn"] = getattr(torch.nn, policy_kwargs["activation_fn"])
    params = dict(config.PPO_PARAMS)
    if continuous:
        policy_kwargs.update(config.CONTINUOUS_ONLY_POLICY_KWARGS)
        params.update(config.CONTINUOUS_ONLY_PARAMS)
    params.update(config.RECIPE_FIXES[recipe])     # the one setting a recipe fix changes
    return PPO(
        "CnnPolicy",
        venv,
        learning_rate=linear_schedule(config.LEARNING_RATE),
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=config.DEVICE,
        tensorboard_log=tensorboard_log,
        verbose=0,
        **params,
    )


def build_beta_ppo(venv, seed, tensorboard_log, recipe):
    """Recipe "beta" (config.py): Beta policy, one pedal, the paper's PPO settings; "beta_slaoui" adds the penalty."""
    policy_kwargs = dict(config.BETA_POLICY_KWARGS)
    policy_kwargs["activation_fn"] = getattr(torch.nn, policy_kwargs["activation_fn"])
    extra = {}
    penalty = config.BETA_RECIPES[recipe]["copy_penalty"]
    if penalty is not None:
        policy_kwargs["features_extractor_class"] = NormalViewCNN
        extra["copy_penalty"] = penalty
    return algorithm_class(recipe)(
        BetaCnnPolicy,
        venv,
        learning_rate=linear_schedule(config.BETA_LEARNING_RATE),
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=config.DEVICE,
        tensorboard_log=tensorboard_log,
        verbose=0,
        **config.BETA_PPO_PARAMS,
        **extra,
    )
