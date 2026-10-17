"""
Train one agent (A or B) with one seed. Safe to stop and restart: it resumes from the last checkpoint.

    python -m scripts.train --agent A --seed 42

Writes to runs/<agent>_seed<seed>/:
    run_info.json          settings, versions, git commit, timing, status
    train_episodes.csv     every training race (score, track covered, ...) = learning curve
    validation.csv         validation races at every checkpoint (normal colors, 20 tracks)
    checkpoints/           model_<decisions>.zip + vecnormalize_<decisions>.pkl, and model_final.zip
    tb/                    TensorBoard logs (losses etc.)
"""

import argparse
import json
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

import config
from racecar.evaluate import episode_record, evaluate, summarize
from racecar.train_utils import build_eval_vec_env, build_ppo, build_vec_env


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def git_state():
    def git(*args):
        return subprocess.run(["git", *args], cwd=config.PROJECT_ROOT, capture_output=True, text=True).stdout.strip()

    return git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))


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
        found = []
        for p in self.ckpt_dir.glob("model_*.zip"):
            tag = p.stem.split("_", 1)[1]
            if tag.isdigit():
                found.append((int(tag), p))
        return max(found) if found else (None, None)

    def save_info(self):
        self.info_path.write_text(json.dumps(self.info, indent=2))

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

    def __init__(self, run, eval_venv, start_timesteps):
        super().__init__()
        self.run, self.eval_venv = run, eval_venv
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
        if self.num_timesteps >= self.next_checkpoint:
            checkpoint(self.run, self.model, self.eval_venv, self.num_timesteps, tag=str(self.num_timesteps))
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


def checkpoint(run, model, eval_venv, timesteps, tag):
    model.save(run.ckpt_dir / f"model_{tag}.zip")
    model.get_vec_normalize_env().save(str(run.ckpt_dir / f"vecnormalize_{tag}.pkl"))
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=sorted(config.AGENTS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--total-timesteps", type=int, default=config.TOTAL_TIMESTEPS)
    p.add_argument("--checkpoint-every", type=int, default=config.CHECKPOINT_EVERY)
    p.add_argument("--val-tracks", type=int, default=len(config.VALIDATION_DURING_TRAINING))
    p.add_argument("--run-dir", default=None, help="default: runs/<agent>_seed<seed> (other folders are for tests)")
    args = p.parse_args()

    torch.set_num_threads(config.TORCH_CPU_THREADS)
    run_dir = config.RUNS_DIR / f"{args.agent}_seed{args.seed}" if args.run_dir is None else config.PROJECT_ROOT / args.run_dir
    run = Run(args.agent, args.seed, args.total_timesteps, args.checkpoint_every, args.val_tracks, run_dir)
    if run.info and run.info["status"] == "complete":
        log(f"{run_dir.name} is already complete. Nothing to do.")
        return

    free_gb = torch.cuda.mem_get_info()[0] / 2**30
    if free_gb < config.MIN_FREE_GPU_GB:
        log(f"STOP: only {free_gb:.2f} GB GPU memory free (need {config.MIN_FREE_GPU_GB}). Not stopping anything; ask the user.")
        sys.exit(3)

    agent_cfg = config.AGENTS[args.agent]
    commit, dirty = git_state()
    if run.info is None:
        run.info = {
            "agent": args.agent,
            "agent_description": agent_cfg["description"],
            "seed": args.seed,
            "color_mode": agent_cfg["color_mode"],
            "continuous_actions": config.CONTINUOUS_ACTIONS,
            "grayscale": config.GRAYSCALE,
            "total_timesteps_target": args.total_timesteps,
            "checkpoint_every": args.checkpoint_every,
            "validation_tracks_during_training": run.val_tracks,
            "hyperparameters_source": "rl_zoo3 2.9.1 hyperparams/ppo.yml, CarRacing-v3 (grayscale switched off: see config.py)",
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
    start_ts, ckpt_path = run.latest_checkpoint()
    session_idx = len(run.info["sessions"])
    env_seed = args.seed if session_idx == 0 else config.RESUME_ENV_SEED_BASE + 1000 * session_idx + args.seed
    run.info["sessions"].append({
        "started": datetime.now().isoformat(timespec="seconds"),
        "resumed_from_timesteps": int(start_ts or 0),
        "env_seed": env_seed,
        "git_commit": commit,
        "git_dirty": dirty,
        "gpu_free_gb_at_start": round(free_gb, 2),
    })
    run.save_info()

    eval_venv = build_eval_vec_env(config.EVAL_N_ENVS, config.CONTINUOUS_ACTIONS, config.GRAYSCALE)
    if final_path.exists():
        log("Training already finished earlier; redoing only the final checkpoint step.")
        venv = build_vec_env(config.N_ENVS, env_seed, agent_cfg["color_mode"], config.CONTINUOUS_ACTIONS, config.GRAYSCALE,
                             vecnormalize_path=run.ckpt_dir / "vecnormalize_final.pkl")
        model = PPO.load(final_path, env=venv, device=config.DEVICE)
    else:
        if ckpt_path is None:
            log(f"Starting {run_dir.name}: agent {args.agent} ({agent_cfg['description']}), seed {args.seed}, "
                f"{args.total_timesteps:,} decisions. GPU free: {free_gb:.2f} GB.")
            venv = build_vec_env(config.N_ENVS, env_seed, agent_cfg["color_mode"], config.CONTINUOUS_ACTIONS, config.GRAYSCALE)
            model = build_ppo(venv, args.seed, config.CONTINUOUS_ACTIONS, tensorboard_log=str(run_dir / "tb"))
            start_ts = 0
        else:
            log(f"RESUMING {run_dir.name} from checkpoint at {start_ts:,} decisions (session {session_idx}).")
            run.drop_rows_after(start_ts)
            run.info["checkpoints"] = [c for c in run.info["checkpoints"] if c["timesteps"] <= start_ts]
            venv = build_vec_env(config.N_ENVS, env_seed, agent_cfg["color_mode"], config.CONTINUOUS_ACTIONS, config.GRAYSCALE,
                                 vecnormalize_path=run.ckpt_dir / f"vecnormalize_{start_ts}.pkl")
            model = PPO.load(ckpt_path, env=venv, device=config.DEVICE)
            set_random_seed(env_seed, using_cuda=True)
        t0 = time.time()
        model.learn(
            total_timesteps=args.total_timesteps - start_ts,
            callback=TrainingLogger(run, eval_venv, start_ts),
            reset_num_timesteps=start_ts == 0,
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
