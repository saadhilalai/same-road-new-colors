# Progress log

Race-car AI that learns to drive from pixels only (Gymnasium CarRacing-v3, PPO),
testing whether domain randomization (random colors in training) makes it robust to unseen colors.

## Step 1: Setup (done, 2026-10-16)

### Software (all pinned in requirements.txt)
- Python 3.11.16 (already installed by `uv`), venv at `.venv`
- PyTorch 2.14.1 (CUDA 13.0 build), Gymnasium 1.4.0, Stable-Baselines3 2.9.0, Box2D 2.3.10
- GPU check (`scripts/step1_check_gpu.py`): CUDA available, NVIDIA GeForce RTX 3080, 10.84 of 12 GB free at the time
  (Ollama was idle; with Ollama loaded, about 6.5 GB free). Training needs under 0.1 GB, so Ollama never needs stopping.
- WSL sees 15.6 GB of RAM (Windows default: half of 32 GB). Training uses about 6.5 GB, so no change needed.

### Environment: what we verified (`scripts/step1_smoke_test.py`, results in `results/step1/`)
- Gymnasium's built-in `domain_randomize=True` draws colors from the same random stream as the track,
  so it changes the track too (verified). Our `TwoSeedCarRacing` (`racecar/envs.py`) gives colors their own seed.
- With normal colors, our env is pixel-for-pixel and reward-for-reward identical to official CarRacing-v3 (verified).
- Changing the color seed changes colors (mean pixel change 28-93 out of 255) and leaves the track unchanged (verified).
- Same seed + same actions = identical run (verified), so every test run can be replayed exactly for videos.
- Random (untrained) car, scratch track 90000: score -31.0, covered 6.9% of 290 tiles, ran out the 1000-frame clock.
- Random-color training env: 8 simulators get different colors, all from the training color range, reproducible from the seed.

### Training settings source
RL Baselines3 Zoo, `rl_zoo3` 2.9.1, `hyperparams/ppo.yml`, entry `CarRacing-v3` (copied into `config.py`):
frame skip 2, resize 64x64, grayscale, stack 2 frames, 8 envs, 4M steps, CnnPolicy, n_steps 512, batch 128,
10 epochs, gamma 0.99, GAE 0.95, clip 0.2, lr linear from 1e-4, ent 0, vf 0.5, max grad norm 0.5,
gSDE (sample freq 4, log_std_init -2), GELU, pi/vf [256], ortho_init False, reward normalization only.
- The Zoo has no discrete-action (5-button) CarRacing recipe. SAC/TQC recipes need a pre-trained autoencoder (not pixels-only).
- Risk: the Zoo's published model with these exact settings scored 175 +/- 100 (sb3/ppo-CarRacing-v0; one run, older v0 env).

### Speed (`scripts/step1_benchmark.py`, `results/step1/benchmark.json`)
| Measurement | Result |
|---|---|
| 1 simulator | 210 frames/s |
| 8 / 10 simulators in lock-step, random actions | 401 / 480 decisions/s |
| PPO training, 8 sims, GPU (continuous+color / continuous+gray / discrete+color / discrete+gray) | 308 / 319 / 311 / 323 decisions/s |
| Hours per 4M-decision run | 3.6 / 3.5 / 3.6 / 3.4 |
| Share of training time: simulators vs GPU learning | 83% vs 17% |
| Evaluation, 10 sims, AI choosing actions | 422 decisions/s |

Bottleneck = the simulator's drawing code on the CPU. 8 independent simulators each run ~27% slower than one alone,
and lock-step stepping costs about another 30%.

### Time budget (planning number: 3.6 h per training run)
| Step | Compute | Estimate |
|---|---|---|
| 2 | Agent A, 3 seeds x 4M decisions + checks + analysis tracks + videos | ~11.5 h (over 8 h: one run at a time, 3.6 h each) |
| 3 | A on unseen colors; Agent B, 3 seeds x 4M; A vs B evaluations | ~11.5 h (over 8 h: one run at a time, 3.6 h each) |
| 4 | Attention maps + heat-map videos | < 1 h |
| 5 | Review | none |
| 6 | Final test: 6 agents x 100 tracks x 2 color conditions | ~0.5 h |
| 7 | Charts + videos | ~1 h (estimate, not measured) |
| 8 | Packaging | none |
| Total | | ~25 h, ~22 h of it training |

### Decisions needed before Step 2
1. AI input: color (recommended) or grayscale (Zoo default).
2. "Lap completed": your strict 100%-of-tiles rule vs Gymnasium's built-in rule (>95% of tiles + crossing the start line).
