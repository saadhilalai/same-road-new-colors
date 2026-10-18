"""
All project settings live in this one file.

Rule: no other file hard-codes a setting. If you want to change something
(seeds, training length, image size...), change it here, and every script
picks it up.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_ROOT / "runs"          # training outputs (big, not in git)
RESULTS_DIR = PROJECT_ROOT / "results"    # small outputs we keep in git

# ---------------------------------------------------------------------------
# Hardware limits (ground rules 3 and 4)
# ---------------------------------------------------------------------------
DEVICE = "cuda"               # train on the RTX 3080
MAX_SIM_PROCESSES = 10        # at most 10 simulators in one training/evaluation
CPU_THREAD_BUDGET = 20        # updated in Step 2: user allows 20 threads in total
MAX_PARALLEL_TRAININGS = 2    # updated in Step 2: two trainings at once, if memory and speed checks pass
MIN_FREE_GPU_GB = 2.0         # refuse to start training with less free GPU memory
MIN_FREE_PC_MEMORY_GB = 3.0   # director's rule (2026-10-17): the PC also runs other apps; below this, stop my jobs
                              # (scripts/memory_guard.py). Measured on Windows ("Available" memory), not just inside WSL.
NICE_LEVEL = 10               # director's rule: run at low CPU priority
TORCH_CPU_THREADS = 2         # keep PyTorch from grabbing every CPU thread

# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
BASE_ENV_ID = "CarRacing-v3"              # the official Gymnasium environment
ENV_ID = "TwoSeedCarRacing-v3"            # our copy: same physics, track and colors seeded separately
MAX_EPISODE_FRAMES = 1000                 # official time limit (50 frames = 1 second)
SOLVED_SCORE = 900                        # official "solved" mark

# Default (normal) colors, copied from Gymnasium's CarRacing source.
NORMAL_ROAD_COLOR = (102, 102, 102)
NORMAL_BG_COLOR = (102, 204, 102)
NORMAL_GRASS_COLOR = (102, 230, 102)

# ---------------------------------------------------------------------------
# What the AI sees (observation processing)
# ---------------------------------------------------------------------------
FRAME_SKIP = 2          # the AI decides every 2nd frame; the action is repeated in between
OBS_SIZE = 64           # shrink the 96x96 image to 64x64
GRAYSCALE = False       # DECIDED (Step 1): keep color. The Zoo uses grayscale; we don't, because the experiment is about colors
FRAME_STACK = 2         # the AI sees its last 2 images, so it can tell speed and direction

# ---------------------------------------------------------------------------
# What the AI controls
# ---------------------------------------------------------------------------
CONTINUOUS_ACTIONS = True   # DECIDED (Step 2): smooth steer/gas/brake (matches the Zoo; 5-button gas is capped at 20%)

# ---------------------------------------------------------------------------
# Metrics (definitions from the project brief)
# ---------------------------------------------------------------------------
# score            = official Gymnasium reward, summed over the episode
# track_covered    = % of track tiles the car drove over
# lap_completed    = covered EVERY tile before time ran out (strict 100% rule, DECIDED Step 1)
# lap_finished_gym = extra column: Gymnasium's own rule (>95% of tiles + touching the start tile)
# off_track        = left the playing area (episode ends, -100)

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
TRAINING_SEEDS = [42, 142, 242]           # 3 seeds for every agent

# Track seeds (which race track gets generated). Kept strictly separate.
VALIDATION_TRACKS = list(range(5000, 5100))     # during training + picking settings
ANALYSIS_TRACK_START = 20000                    # comparisons along the way
ANALYSIS_TRACKS = list(range(20000, 20100))     # 100 analysis tracks
TEST_TRACKS = list(range(10000, 10100))         # HIDDEN: used exactly once, in Step 6
SCRATCH_TRACK_START = 90000                     # smoke tests and demos only
FILLER_TRACK = 99999                            # evaluation simulators with no work left drive here (ignored)

# Training simulators get their own track seeds: run seed + simulator number (42..49, 142..149, 242..249).
# If a run is resumed, the restarted simulators get seeds far away from every evaluation range.
RESUME_ENV_SEED_BASE = 3_000_000

# Color seeds (which random color scheme gets used).
# Colors come from their own random-number stream, separate from the track.
# Agent B only ever trains on color seeds inside this range...
TRAIN_COLOR_SEED_RANGE = (1_000_000, 2_000_000)
# ...and every evaluation uses color seeds below 1,000,000, so by construction
# no agent has ever trained on an evaluation color scheme.
COLOR_STREAM_ID = 7   # mixed into the color seed so color numbers never reuse the track's random stream
# Evaluation with unseen colors: race on track T uses color seed T (same number, separate random stream,
# so the colors have nothing to do with the track's shape). Validation colors 5000-5099, analysis colors
# 20000-20099, test colors 10000-10099: all below 1,000,000, so never seen in training.

# ---------------------------------------------------------------------------
# Training settings: RL Baselines3 Zoo, rl_zoo3 v2.9.1, hyperparams/ppo.yml, "CarRacing-v3"
# ---------------------------------------------------------------------------
ALGORITHM = "PPO"
N_ENVS = 8                       # simulators running in parallel during training
TOTAL_TIMESTEPS = 4_000_000      # AI decisions per training run (= 8M simulator frames with frame skip 2)
VEC_NORMALIZE = dict(norm_obs=False, norm_reward=True)   # rescale rewards only, not images
PPO_PARAMS = dict(
    n_steps=512,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    n_epochs=10,
    ent_coef=0.0,
    max_grad_norm=0.5,
    vf_coef=0.5,
    clip_range=0.2,
)
LEARNING_RATE = 1e-4             # Zoo "lin_1e-4": starts at 1e-4, shrinks linearly to 0
CONTINUOUS_ONLY_PARAMS = dict(use_sde=True, sde_sample_freq=4)    # gSDE exploration (continuous only)
POLICY_KWARGS = dict(
    ortho_init=False,
    activation_fn="GELU",
    net_arch=dict(pi=[256], vf=[256]),
)
CONTINUOUS_ONLY_POLICY_KWARGS = dict(log_std_init=-2)

# Recipe fixes (Step 2). The Zoo recipe collapsed in the seed-42 pilot (PROGRESS.md, Step 2): its exploration
# noise shrank to almost nothing and the driving got worse. Each fix changes exactly ONE PPO setting.
# Both were tried on seed 42 side by side; the winner was picked on validation tracks only.
RECIPE_FIXES = {
    "zoo": {},                                   # the Zoo recipe unchanged (the paused pilot)
    "entropy": dict(ent_coef=0.01),              # entropy bonus: a small reward for keeping some randomness
    "kl_limit": dict(target_kl=0.05),            # end a training round early once the driver has changed too much
}
RECIPE = "entropy"                               # DECIDED 2026-10-17: trial winner on validation (results/step2/recipe_trial.md)
RECIPE_TRIAL_MIN_SCORE = 500                     # if neither fix's final model reaches this on validation: stop


def run_name(agent, seed, recipe=None):
    """Folder name of one training run. The Zoo recipe has no suffix (the pilot); a fix adds its name."""
    recipe = RECIPE if recipe is None else recipe
    return f"{agent}_seed{seed}" + ("" if recipe == "zoo" else f"_{recipe}")

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
AGENTS = {
    "A": dict(color_mode="normal", description="trained on normal colors only"),
    "B": dict(color_mode="random", description="trained with random colors every episode (domain randomization)"),
}

# ---------------------------------------------------------------------------
# Checkpoints and checks during training
# ---------------------------------------------------------------------------
CHECKPOINT_EVERY = 250_000                       # save + validate every 250k decisions (and at 0 and the end)
VALIDATION_DURING_TRAINING = VALIDATION_TRACKS[:20]   # 20 tracks, normal colors
EVAL_N_ENVS = 8                                  # simulators used for evaluation (training sims sit idle meanwhile)
PILOT_PASS_SCORE = 800                           # seed 42 must average this on all 100 validation tracks
AGENT_CHECKPOINT = "final"                       # DECIDED: each run's final copy is the agent (A and B alike); no picking
