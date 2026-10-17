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
MAX_SIM_PROCESSES = 10        # at most 10 simulators running at once
MIN_FREE_GPU_GB = 2.0         # refuse to start training with less free GPU memory
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
GRAYSCALE = None        # DECISION PENDING (end of Step 1): True = grayscale (Zoo), False = keep color
FRAME_STACK = 2         # the AI sees its last 2 images, so it can tell speed and direction

# ---------------------------------------------------------------------------
# What the AI controls
# ---------------------------------------------------------------------------
CONTINUOUS_ACTIONS = None   # DECISION PENDING (Step 2): True = smooth steer/gas/brake, False = 5 buttons

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------
TRAINING_SEEDS = [42, 142, 242]           # 3 seeds for every agent

# Track seeds (which race track gets generated). Kept strictly separate.
VALIDATION_TRACKS = list(range(5000, 5100))     # during training + picking settings
ANALYSIS_TRACK_START = 20000                    # comparisons along the way (20000, 20001, ...)
TEST_TRACKS = list(range(10000, 10100))         # HIDDEN: used exactly once, in Step 6
SCRATCH_TRACK_START = 90000                     # smoke tests and demos only

# Color seeds (which random color scheme gets used).
# Colors come from their own random-number stream, separate from the track.
# Agent B only ever trains on color seeds inside this range...
TRAIN_COLOR_SEED_RANGE = (1_000_000, 2_000_000)
# ...and every evaluation uses color seeds below 1,000,000, so by construction
# no agent has ever trained on an evaluation color scheme.
COLOR_STREAM_ID = 7   # mixed into the color seed so color numbers never reuse the track's random stream

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
