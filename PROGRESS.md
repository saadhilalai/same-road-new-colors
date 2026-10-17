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

## Step 2: First driver (in progress)

### Decisions (from the director)
- AI input: color. Controls: continuous (steer, gas, brake), as in the Zoo recipe.
- "Lap completed" = strict 100% of tiles; Gymnasium's own rule is reported as an extra column (`lap_finished_gym`).
- The agent is each run's **final** copy (after all 4M decisions), for A and B alike (`config.AGENT_CHECKPOINT`).
- If seed 42 doesn't drive well: show what went wrong before changing anything; fix using validation tracks only.

### Pilot run: Agent A, seed 42, Zoo recipe unchanged (started 2026-10-16 21:32)
Learned fast, then got worse. Validation = 20 validation tracks at each checkpoint (`runs/A_seed42/validation.csv`).

| Decisions | Validation mean score | Track covered | Laps | Off track |
|---|---|---|---|---|
| 0 | -70.0 | 5.3% | 0% | 80% |
| 254k | 674.7 | 77.5% | 0% | 0% |
| 504k | 742.7 | 85.0% | 15% | 10% |
| 754k | 738.2 | 84.4% | 20% | 10% |
| 1.00M | 556.7 | 68.6% | 0% | 35% |
| 1.25M | 388.2 | 52.1% | 0% | 45% |

Checked on all 100 validation tracks (`results/evals/A_seed42/validation_normal_ckpt*.csv`), every race replayed exactly (100/100):

| Checkpoint | Mean score | Covered | Laps (strict) | Off track | Ran out of time | All wheels on grass |
|---|---|---|---|---|---|---|
| 504k (best so far) | 772.8 +/- 184.3 | 87.4% | 18% | 4% | 78% | 9.5% of the time |
| 1.25M | 348.5 +/- 191.7 | 47.6% | 0% | 40% | 60% | 46.2% of the time |

What the driving looks like (`results/driving/A_seed42_validation_normal_ckpt*/`):
- **504k:** smooth, stays on the road, no tricks seen (no shortcuts, almost no driving backwards). **Slow:** average gas 0.16 of max,
  never full gas. 50 of 100 races ran out of time with 90%+ of the track already covered.
  Worst race: misses a sharp hairpin, drives straight across the grass and off the map without trying to come back.
- **1.25M:** the same failure, much more often. It steers less (average |steer| 0.049 vs 0.078) and goes straight
  at corners; once the road is out of view it drives straight until it leaves the playing area.

Training health (`results/step2/A_seed42_training_health.png`, from TensorBoard):
- The exploration noise ("std", how much random trying is left in its actions) fell steadily: 0.135 at the start, 0.017 at 1.4M.
- How much each training round changed the driver ("approx KL") rose from about 0.05 to about 0.85.
  Spinning Up's PPO notes call 0.01-0.05 the usual size. About 75% of training samples hit PPO's clip limit.
- Likely reason (not yet tested): the exploration noise collapses (nothing in the recipe keeps it up: ent_coef = 0),
  so each update becomes erratic and the driver drifts into under-steering. The Zoo's published result for this
  recipe (175 +/- 100, CarRacing-v0, one run) fits the same picture, but that is also untested.

### Director's decision (2026-10-17, ~00:40): option A, test two fixes side by side
The pilot was paused at 2.85M decisions (resumable; last checkpoint 2,752,512). It had not recovered:
validation (20 tracks) at 1.5M-2.75M stayed between 247 and 369. Pilot status in `runs/A_seed42/run_info.json`: "paused".

Two new seed-42 runs, trained at the same time, each the Zoo recipe with exactly ONE change (`config.RECIPE_FIXES`):

| Recipe | Change | Plain English | Run folder |
|---|---|---|---|
| `entropy` | `ent_coef` 0.0 -> 0.01 | a small reward for keeping some randomness, so exploration can't collapse to zero | `runs/A_seed42_entropy` |
| `kl_limit` | `target_kl` None -> 0.05 | each training round stops early once the driver has changed too much (SB3 stops at 1.5 x 0.05) | `runs/A_seed42_kl_limit` |

Why these values: 0.01 is the most common PPO entropy setting; 0.05 is the upper end of the "usual" KL range
in OpenAI Spinning Up's PPO notes ("usually small, 0.01 or 0.05"). Neither was tuned. Same seed (42) and same
training tracks as the pilot, so the only difference between the runs is the one setting.

**Selection rule (written down before any result):**
1. Each trial's agent = its final copy (after 4M decisions), as decided.
2. Evaluate both final copies on all 100 validation tracks, normal colors. Test tracks are not touched.
3. Winner = higher mean validation score. A paired test across the 100 tracks (Wilcoxon signed-rank) is
   reported too; if it is not significant, the pick is still the higher mean, and that is said plainly.
4. The winning recipe becomes `config.RECIPE` for A and for B (Step 3); its seed-42 run counts as A seed 42.
   Seeds 142 and 242 then train together with it, without waiting for the director.
5. Stop and wait for the director only if something is truly broken: a run crashes and can't resume, or
   neither fix's final copy reaches a validation mean of 500 (`config.RECIPE_TRIAL_MIN_SCORE`), i.e. both
   still collapse. Nothing else changes (for example the car's slowness is reported, not fixed).

Small honesty note: picking the recipe with seed 42's validation result and then also reporting seed 42 is a
mild selection effect on validation tracks only; seeds 142/242 and the hidden test tracks are not affected.
