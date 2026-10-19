"""
Train one agent (A or B) with one seed. Safe to stop and restart: it resumes from the last checkpoint.

Exact resumes (added 2026-10-17, after a Windows Update restart): every checkpoint also saves the full
training state (random-number generators, each simulator's current race, frame stack, reward scaler),
so a resumed run continues bit-for-bit as if it had never stopped. Training is repeatable from the
seed on this PC (tested), and all files are written atomically so a power cut can't leave half a checkpoint.

    python -m scripts.train --agent A --seed 42 [--recipe beta]

Writes to runs/<run name>/ (see config.run_name: e.g. A_seed42 for the Zoo recipe, A_seed42_entropy for a fix):
    run_info.json          settings, versions, git commit, timing, status
    train_episodes.csv     every training race (score, track covered, ...) = learning curve
    validation.csv         validation races at every checkpoint (normal colors, 20 tracks)
    checkpoints/           model_<decisions>.zip + vecnormalize_<decisions>.pkl + runstate_<decisions>.pkl,
                           and model_final.zip
    tb/                    TensorBoard logs (losses etc.)
"""

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
import time
from collections import deque
from datetime import datetime

import gymnasium
import numpy as np
import pandas as pd
import stable_baselines3
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

import config
from racecar.evaluate import episode_record, evaluate, summarize
from racecar.train_utils import algorithm_class, build_eval_vec_env, build_ppo, build_vec_env


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


def atomic_write(path, write):
    """write(tmp_path), then rename: the file is either the old version or the complete new one, never half."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def config_snapshot():
    """Every setting from config.py, minus folder paths (they would contain the user's home folder)."""
    snap = {}
    for k, v in vars(config).items():
        if k.isupper() and not hasattr(v, "resolve"):
            snap[k] = v
    return json.loads(json.dumps(snap, default=str))


class Run:
    """Everything about one training run's folder."""

    def __init__(self, agent, seed, total, every, n_val, run_dir):
        self.agent, self.seed, self.total, self.every = agent, seed, total, every
        self.dir = run_dir
        self.ckpt_dir = run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.info_path = run_dir / "run_info.json"
        self.val_tracks = config.VALIDATION_DURING_TRAINING[:n_val]
        self.info = json.loads(self.info_path.read_text()) if self.info_path.exists() else None

    def latest_checkpoint(self):
        """
        Newest complete checkpoint: (decisions, model path, full training state or None).
        A run state is written last, so if a run has any, only checkpoints with one count as complete.
        Older runs (weights only) fall back to their newest weights: an approximate resume.
        """
        found = {}
        for p in self.ckpt_dir.glob("model_*.zip"):
            tag = p.stem.split("_", 1)[1]
            if tag.isdigit() and (self.ckpt_dir / f"vecnormalize_{tag}.pkl").exists():
                found[int(tag)] = p
        full = [ts for ts in found if (self.ckpt_dir / f"runstate_{ts}.pkl").exists()]
        if full:
            ts = max(full)
            with open(self.ckpt_dir / f"runstate_{ts}.pkl", "rb") as f:
                return ts, found[ts], pickle.load(f)
        if found:
            ts = max(found)
            return ts, found[ts], None
        return None, None, None

    def save_info(self):
        atomic_write(self.info_path, lambda tmp: tmp.write_text(json.dumps(self.info, indent=2)))

    def drop_validation_at(self, timesteps):
        path = self.dir / "validation.csv"
        if path.exists():
            df = pd.read_csv(path)
            df[df["timesteps"] != timesteps].to_csv(path, index=False)

    def drop_rows_after(self, timesteps):
        """On resume: forget log rows written after the checkpoint we roll back to."""
        for name in ("train_episodes.csv", "validation.csv"):
            path = self.dir / name
            if path.exists():
                df = pd.read_csv(path)
                df[df["timesteps"] <= timesteps].to_csv(path, index=False)

    def append_csv(self, name, rows):
        path = self.dir / name
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


class TrainingLogger(BaseCallback):
    """Logs every finished training race; saves + validates every CHECKPOINT_EVERY decisions."""

    def __init__(self, run, eval_venv, start_timesteps, restore_state=None):
        super().__init__()
        self.run, self.eval_venv = run, eval_venv
        self.restore_state = restore_state
        self.next_checkpoint = 0 if start_timesteps == 0 else (start_timesteps // run.every + 1) * run.every
        self.session_start = (time.time(), start_timesteps)
        self.recent = deque(maxlen=100)
        self.buffer = []
        self.rollouts = 0

    def _on_step(self):
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if done and "episode" in info:
                row = episode_record(info)
                row.update(timesteps=self.num_timesteps, env=i)
                self.buffer.append(row)
                self.recent.append(row["score"])
        return True

    def _on_rollout_start(self):
        # Runs right after each learning update, so a checkpoint here holds a clean state.
        if self.restore_state is not None:
            restore_run_state(self.model, self, self.restore_state)
            log(f"Exact resume: restored the full training state at {self.num_timesteps:,} decisions "
                f"({len(self.restore_state['env_states'])} simulators rebuilt and checked against the saved images).")
            self.restore_state = None
            self.next_checkpoint = (self.num_timesteps // self.run.every + 1) * self.run.every
            return
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint(self.run, self.model, self.eval_venv, self.num_timesteps, tag=str(self.num_timesteps), logger=self)
            self.next_checkpoint = (self.num_timesteps // self.run.every + 1) * self.run.every

    def _on_rollout_end(self):
        self.run.append_csv("train_episodes.csv", self.buffer) if self.buffer else None
        self.buffer = []
        self.rollouts += 1
        if self.rollouts % 12 == 0:   # about every 50k decisions
            t0, s0 = self.session_start
            speed = (self.num_timesteps - s0) / (time.time() - t0)
            eta_h = (self.run.total - self.num_timesteps) / speed / 3600
            mean100 = np.mean(self.recent) if self.recent else float("nan")
            log(f"{self.num_timesteps:>9,} decisions ({100 * self.num_timesteps / self.run.total:5.1f}%) | "
                f"last-100 training races mean score {mean100:7.1f} | {speed:5.0f} decisions/s | ETA {eta_h:4.2f} h")


def frame_stack_of(venv):
    while not isinstance(venv, VecFrameStack):
        venv = venv.venv
    return venv


def capture_run_state(model, logger):
    """Everything besides the weights that the next training step depends on (for exact resumes)."""
    venv = model.get_env()
    return {
        "timesteps": int(model.num_timesteps),
        "env_states": venv.env_method("get_resume_state"),
        "stacked_obs": frame_stack_of(venv).stacked_obs.stacked_obs.copy(),
        "vecnormalize_returns": model.get_vec_normalize_env().returns.copy(),
        "last_obs": model._last_obs.copy(),
        "last_episode_starts": model._last_episode_starts.copy(),
        "ep_info_buffer": list(model.ep_info_buffer),
        "ep_success_buffer": list(model.ep_success_buffer),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
        "logger": {"recent": list(logger.recent), "rollouts": logger.rollouts},
    }


def restore_run_state(model, logger, state):
    """Undo capture_run_state. Each simulator is rebuilt by replaying its current race, then checked."""
    venv = model.get_env()
    for i, env_state in enumerate(state["env_states"]):
        obs = venv.env_method("set_resume_state", env_state, indices=i)[0]
        newest = state["stacked_obs"][i][..., -obs.shape[-1]:]
        if not np.array_equal(obs, newest):
            raise RuntimeError(f"Exact resume failed: simulator {i} does not match its saved image.")
    stack = frame_stack_of(venv).stacked_obs
    stack.stacked_obs = state["stacked_obs"].copy()
    model.get_vec_normalize_env().returns = state["vecnormalize_returns"].copy()
    # Rebuild the AI's last image exactly as the wrappers hand it over: a transposed VIEW, not a tidy copy.
    # Its memory layout decides which (equally correct) GPU method runs, and the two differ in the last
    # digits (measured: 2.7e-5), which is enough to make a resumed run drift away from an uninterrupted one.
    last_obs = venv.transpose_image(stack.stacked_obs) if isinstance(venv, VecTransposeImage) else stack.stacked_obs
    if not np.array_equal(last_obs, state["last_obs"]):
        raise RuntimeError("Exact resume failed: rebuilt observation does not match the saved one.")
    model._last_obs = last_obs
    model._last_episode_starts = state["last_episode_starts"].copy()
    model.ep_info_buffer = deque(state["ep_info_buffer"], maxlen=model.ep_info_buffer.maxlen)
    model.ep_success_buffer = deque(state["ep_success_buffer"], maxlen=model.ep_success_buffer.maxlen)
    logger.recent = deque(state["logger"]["recent"], maxlen=100)
    logger.rollouts = state["logger"]["rollouts"]
    random.setstate(state["rng"]["python"])
    np.random.set_state(state["rng"]["numpy"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state_all(state["rng"]["cuda"])


def checkpoint(run, model, eval_venv, timesteps, tag, logger=None):
    atomic_write(run.ckpt_dir / f"model_{tag}.zip", model.save)
    atomic_write(run.ckpt_dir / f"vecnormalize_{tag}.pkl", lambda tmp: model.get_vec_normalize_env().save(str(tmp)))
    df = evaluate(model, [(t, None) for t in run.val_tracks], venv=eval_venv)
    df.insert(0, "timesteps", timesteps)
    run.append_csv("validation.csv", df.to_dict("records"))
    s = summarize(df)
    log(f"CHECKPOINT {tag}: validation ({s['races']} tracks, normal colors) mean score {s['mean_score']} "
        f"+/- {s['std_score']} | covered {s['mean_track_covered_pct']}% | laps {s['lap_completed_pct']}% "
        f"| off track {s['off_track_pct']}%")
    run.info["timesteps_done"] = int(timesteps)
    run.info["checkpoints"].append({"tag": tag, "timesteps": int(timesteps), "time": datetime.now().isoformat(timespec="seconds"), **s})
    run.save_info()
    if logger is not None:   # written last: its presence marks the checkpoint as complete
        state = capture_run_state(model, logger)
        atomic_write(run.ckpt_dir / f"runstate_{tag}.pkl", lambda tmp: tmp.write_bytes(pickle.dumps(state)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=sorted(config.AGENTS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, default=None, help="default: config.total_timesteps(recipe)")
    p.add_argument("--checkpoint-every", type=int, default=config.CHECKPOINT_EVERY)
    p.add_argument("--val-tracks", type=int, default=len(config.VALIDATION_DURING_TRAINING))
    p.add_argument("--recipe", choices=config.ALL_RECIPES, default=config.RECIPE)
    p.add_argument("--run-dir", default=None, help="default: runs/<run name> (other folders are for tests)")
    args = p.parse_args()
    if args.total_timesteps is None:
        args.total_timesteps = config.total_timesteps(args.recipe)

    torch.set_num_threads(config.TORCH_CPU_THREADS)
    # Same GPU math every run (evaluate() also sets this). With it, training repeats bit-for-bit from the seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    run_name = config.run_name(args.agent, args.seed, args.recipe)
    run_dir = config.RUNS_DIR / run_name if args.run_dir is None else config.PROJECT_ROOT / args.run_dir
    run = Run(args.agent, args.seed, args.total_timesteps, args.checkpoint_every, args.val_tracks, run_dir)
    if run.info and run.info.get("recipe", "zoo") != args.recipe:
        log(f"STOP: {run_dir.name} was trained with recipe '{run.info.get('recipe', 'zoo')}', not '{args.recipe}'.")
        sys.exit(2)
    if run.info and run.info["status"] == "complete":
        log(f"{run_dir.name} is already complete. Nothing to do.")
        return

    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    if free_gb < config.MIN_FREE_GPU_GB:
        log(f"STOP: only {free_gb:.2f} GB GPU memory free (need {config.MIN_FREE_GPU_GB}). Not stopping anything; ask the user.")
        sys.exit(3)

    agent_cfg = config.AGENTS[args.agent]
    view = config.recipe_view(args.recipe)
    # Slaoui's method (Agent B's fallback): the races are in normal colors; the random colors go into the copies.
    color_mode = "normal" if view["copies"] else agent_cfg["color_mode"]
    if view["copies"] and agent_cfg["color_mode"] != "random":
        log(f"STOP: recipe '{args.recipe}' is Agent B's fallback (random colors as recolored copies), not for agent {args.agent}.")
        sys.exit(2)
    if args.recipe in config.BETA_RECIPES:
        recipe_settings = {"source": "Petrazzini & Antonelo 2021 (arXiv 2111.02202), Table II; see config.py",
                           "ppo": config.BETA_PPO_PARAMS, "learning_rate": f"{config.BETA_LEARNING_RATE}, linear to 0",
                           "policy": "Beta", "policy_kwargs": config.BETA_POLICY_KWARGS, "one_pedal": view["one_pedal"],
                           "slaoui_copy_penalty": config.BETA_RECIPES[args.recipe]["copy_penalty"]}
    else:
        recipe_settings = {"source": "rl_zoo3 2.9.1 hyperparams/ppo.yml, CarRacing-v3 (grayscale switched off: see config.py)",
                           "change_from_zoo": config.RECIPE_FIXES[args.recipe]}
    commit, dirty = git_state()
    if run.info is None:
        run.info = {
            "agent": args.agent,
            "agent_description": agent_cfg["description"],
            "seed": args.seed,
            "color_mode": color_mode,
            "recolored_copies": "random colors (B's color range), one scheme per race" if view["copies"] else None,
            "continuous_actions": config.CONTINUOUS_ACTIONS,
            "grayscale": config.GRAYSCALE,
            "total_timesteps_target": args.total_timesteps,
            "checkpoint_every": args.checkpoint_every,
            "validation_tracks_during_training": run.val_tracks,
            "recipe": args.recipe,
            "recipe_settings": recipe_settings,
            "status": "running",
            "timesteps_done": 0,
            "sessions": [],
            "checkpoints": [],
            "versions": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "gymnasium": gymnasium.__version__,
                "stable_baselines3": stable_baselines3.__version__,
                "numpy": np.__version__,
            },
            "gpu": torch.cuda.get_device_name(0),
            "config": config_snapshot(),
        }

    final_path = run.ckpt_dir / "model_final.zip"
    final_ready = final_path.exists() and (run.ckpt_dir / "vecnormalize_final.pkl").exists()
    start_ts, ckpt_path, saved_state = run.latest_checkpoint()
    exact = saved_state is not None
    session_idx = len(run.info["sessions"])
    # Exact resumes overwrite every random-number state, so they keep the run's own seed.
    env_seed = args.seed if (session_idx == 0 or exact) else config.RESUME_ENV_SEED_BASE + 1000 * session_idx + args.seed
    run.info["sessions"].append({
        "started": datetime.now().isoformat(timespec="seconds"),
        "resumed_from_timesteps": int(start_ts or 0),
        "resume_mode": "new run" if start_ts is None else ("exact (full state restored)" if exact else "approximate (weights only)"),
        "env_seed": env_seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu_free_gb_at_start": round(free_gb, 2),
    })
    run.save_info()

    eval_venv = build_eval_vec_env(config.EVAL_N_ENVS, config.CONTINUOUS_ACTIONS, config.GRAYSCALE, **view)
    algo = algorithm_class(args.recipe)
    if final_ready:
        log("Training already finished earlier; redoing only the final checkpoint step.")
        venv = build_vec_env(config.N_ENVS, env_seed, color_mode, config.CONTINUOUS_ACTIONS, config.GRAYSCALE,
                             vecnormalize_path=run.ckpt_dir / "vecnormalize_final.pkl", **view)
        model = algo.load(final_path, env=venv, device=config.DEVICE)
        run.drop_validation_at(model.num_timesteps)
        run.info["checkpoints"] = [c for c in run.info["checkpoints"] if c["tag"] != "final"]
    else:
        if ckpt_path is None:
            log(f"Starting {run_dir.name}: agent {args.agent} ({agent_cfg['description']}), seed {args.seed}, "
                f"recipe '{args.recipe}', race colors {color_mode}"
                f"{' + random recolored copies (Slaoui)' if view['copies'] else ''}, {args.total_timesteps:,} decisions. "
                f"GPU free: {free_gb:.2f} GB.")
            venv = build_vec_env(config.N_ENVS, env_seed, color_mode, config.CONTINUOUS_ACTIONS, config.GRAYSCALE, **view)
            model = build_ppo(venv, args.seed, config.CONTINUOUS_ACTIONS, tensorboard_log=str(run_dir / "tb"),
                              recipe=args.recipe)
            start_ts = 0
        else:
            log(f"RESUMING {run_dir.name} from checkpoint at {start_ts:,} decisions (session {session_idx}, "
                f"{'exact: full state saved' if exact else 'approximate: weights only'}).")
            run.drop_rows_after(start_ts)
            run.info["checkpoints"] = [c for c in run.info["checkpoints"] if c["timesteps"] <= start_ts]
            venv = build_vec_env(config.N_ENVS, env_seed, color_mode, config.CONTINUOUS_ACTIONS, config.GRAYSCALE,
                                 vecnormalize_path=run.ckpt_dir / f"vecnormalize_{start_ts}.pkl", **view)
            model = algo.load(ckpt_path, env=venv, device=config.DEVICE)
            if not exact:
                set_random_seed(env_seed, using_cuda=True)
        t0 = time.time()
        model.learn(
            total_timesteps=args.total_timesteps - start_ts,
            callback=TrainingLogger(run, eval_venv, start_ts, restore_state=saved_state),
            reset_num_timesteps=ckpt_path is None,
            tb_log_name="PPO",
        )
        run.info["sessions"][-1]["training_hours"] = round((time.time() - t0) / 3600, 3)
        run.save_info()

    checkpoint(run, model, eval_venv, model.num_timesteps, tag="final")
    run.info["status"] = "complete"
    run.info["finished"] = datetime.now().isoformat(timespec="seconds")
    run.info["training_hours_total"] = round(sum(s.get("training_hours", 0) for s in run.info["sessions"]), 3)
    run.save_info()
    log(f"DONE {run_dir.name}: {model.num_timesteps:,} decisions, {run.info['training_hours_total']} h of training.")
    venv.close()
    eval_venv.close()


if __name__ == "__main__":
    main()
