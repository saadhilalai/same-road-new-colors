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

### Interruption: Windows Update restart (2026-10-17, 01:59)
**What happened.** Windows Update restarted the PC at 01:59:29 (Windows event log: event 1074 from
`MoUsoCoreWorker.exe`, "Operating System: Service pack (Planned)", then a clean shutdown; "the last shutdown's
success status was true"). A second update restart followed at 02:02. It was **not** heat or a power problem.
Windows "active hours" are set to 11:00-01:00, so Windows restarts for updates right after 1 AM.
WSL only came back when the director logged in at 12:56, so about 11 hours of machine time were lost.
(Also in the event log: one NVIDIA driver message, event 13 "Graphics FECS Exception: Logging error 0x2", at 23:23
the evening before. Nothing was affected: the pilot kept training and every evaluation and replay matched.)

**Where the runs stopped.** All files were intact (every checkpoint zip and pickle loads; no half-written CSV rows).

| Run | Last logged | Last checkpoint | Training lost |
|---|---|---|---|
| `A_seed42_entropy` | 625,808 decisions | 503,808 | ~14 min |
| `A_seed42_kl_limit` | 736,856 decisions | 503,808 | ~22 min |
| `A_seed42` (Zoo pilot) | paused on purpose earlier | 2,752,512 | none |

Both 503,808 checkpoints were re-validated on their 20 validation tracks: identical to the logged scores, race by race.

**Why not simply resume from 503,808.** Those checkpoints held the AI's weights, its optimizer and the reward
scaler, but not the random-number generators or the 8 simulators' half-finished races. A resume would have been
statistically fine but not identical to an uninterrupted run. A test showed that training on this PC *is*
bit-for-bit repeatable from the seed (two identical 8,192-decision runs: 0 of 33 weight tensors differ), so the
director chose **replaying both trials from the start** (about 1 extra hour), which gives exactly the
uninterrupted result. The interrupted folders are kept in `runs/_archive/` (not in git).

**Upgrade: exact resumes from now on** (`scripts/train.py`, `racecar/envs.py: ResumableEnv`).
- Every checkpoint also saves `runstate_<decisions>.pkl`: all random-number states (Python, NumPy, PyTorch CPU and
  GPU), each simulator's current race (its random state when the race began + every action since), the
  2-frame stack, the reward scaler's running returns, and the AI's last image.
- Resuming rebuilds each simulator by replaying its race and checks it against the saved image.
- Every checkpoint file is written atomically (temp file, then rename), so a power cut can't leave half a file;
  a checkpoint only counts once its run state (written last) exists.
- **Test:** a run hard-killed (`kill -9`) in the middle of a rollout and then resumed was compared with the same run
  left alone: weights, optimizer, all 48 training races and all validation races **identical**. Same result for
  agent B (random colors).
- Bug found by that test and fixed: the AI's last image has to be restored as the same "transposed view" the
  wrappers produce. A tidy copy has a different memory layout, the GPU then picks a different (equally correct)
  calculation method, and the results differ in the 5th decimal (measured 2.7e-5). That is enough for RL training
  to drift apart within one update.
- `scripts/log_resources.sh` now lives in the repo (the old copy was in /tmp and was wiped by the restart).

**Replay check: passed (2026-10-17 16:18).** Each replayed trial reproduced its interrupted original exactly:

| Trial | Checkpoints compared (weights + optimizer) | Training races compared | Validation races | Result |
|---|---|---|---|---|
| `entropy` | 0, 253,952, 503,808: 0 of 33 tensors differ | 1,310 (up to 625,808 decisions) | 60 | identical |
| `kl_limit` | 0, 253,952, 503,808: 0 of 33 tensors differ | 1,559 (up to 736,856 decisions) | 60 | identical |

So the trial results are exactly what they would have been without the restart.

**Advice for the director:** to avoid this during long runs, pause Windows updates (Settings > Windows Update >
Pause updates) or widen active hours. An interruption now costs at most the ~25 min since the last checkpoint,
but nothing restarts by itself until someone logs in.

### Recipe trial result (2026-10-17 21:25): the entropy bonus wins
Both trials ran the full 4M decisions on seed 42 (`entropy` 6.08 h, `kl_limit` 5.69 h of training, two at once).
**Neither collapsed** the way the Zoo pilot did: their 20-track validation checks stayed between 662 and 862 the whole
way (pilot: down to 247). Chart: `results/step2/A_seed42_training_health.png`.

Final copies on all 100 validation tracks (`results/step2/recipe_trial.md`):

| Seed 42 | Mean score | Median | Covered | Laps (strict) | Off track | Out of time | Avg gas | Avg abs(steer) |
|---|---|---|---|---|---|---|---|---|
| Zoo pilot, best checkpoint (504k), for reference | 772.8 | 862.8 | 87.4% | 18% | 4% | 78% | 0.16 | 0.078 |
| Zoo pilot at 1.25M (collapsed), for reference | 348.5 | 331.4 | 47.6% | 0% | 40% | 60% | 0.14 | 0.049 |
| **`entropy` final** | **781.8** | 862.4 | 88.5% | 16% | 6% | 78% | 0.09 | 0.060 |
| `kl_limit` final | 692.7 | 811.8 | 79.1% | 8% | 0% | 92% | 0.25 | 0.280 |

- Paired test on the same 100 tracks: `entropy` better on 66, `kl_limit` on 34; mean difference +89.2;
  Wilcoxon signed-rank p = 0.0005. **By the pre-written rule the winner is `entropy`**, now `config.RECIPE`.
- What that p-value does and doesn't say: it compares these two *trained drivers* across tracks. Each recipe was
  trained once, so it does **not** show that the recipe is better in general (run-to-run variation is unmeasured).
- **My earlier "likely reason" is not supported.** I guessed the pilot collapsed because its exploration noise
  shrank to almost nothing. But the `entropy` run's noise shrank almost as much (from 0.135 to 0.012 at 2.9M decisions,
  vs the pilot's 0.008 at the same point; 0.010 at the end), and its update size (approx KL) was just as large, yet it did not collapse.
  The `kl_limit` run kept its noise high (0.094 at the end) and never left the playing area once trained, but its
  final copy was weaker than its own earlier checkpoints (20-track checks peaked at 861 at 2.75M; final 678).
  With one run per recipe I can't say *why* the pilot collapsed and the `entropy` run didn't; the three A seeds
  (and three B seeds) will show whether collapses still happen with this recipe.
- The pass mark I set for seed 42 (800 on the 100 validation tracks) is **not met**: 781.8. As decided, the
  slowness is reported, not fixed: 55 of 100 races ran out of time with 90%+ of the track covered, and the car
  never uses full gas (average gas 0.09).
- Seed 42's run for Agent A is now `runs/A_seed42_entropy` (its exact replay after the restart is verified above).

### Seeds 142 and 242 (started 2026-10-17 21:28, together)
Recipe `entropy`, launched right after the trial as the director instructed. At checkpoint 0 both score -92.9:
different starting weights (32 of 33 weight tensors differ; the shared one is the fixed starting noise level),
but both untrained cars stand still for all 1,000 frames, which gives the same score.

### Night of 2026-10-17: memory rule, one training at a time
The director's rules for the night: run at low priority, and if free memory drops under 3 GB, pause my work
until it recovers (the PC also runs other apps).
- At 23:13 Windows had only **2.4 GB available** (Task Manager's "Available", which includes the standby cache;
  48.6 of 51.7 GB committed) with my two trainings running. The WSL VM, mostly my two trainings, took 7.9 GB.
  The rest was other apps (several node processes, pythonw, Chrome, ChatGPT).
- A frozen (SIGSTOP) process keeps its memory, so "pausing" has to mean stopping a training. With exact resumes
  that costs only the time since its last checkpoint. **Seed 242 was stopped at 23:15** (about 50k decisions after
  its 1,003,520 checkpoint). Windows available memory went from 2.4 GB to 4.2 GB, so one training costs about
  2 GB on the Windows side.
- So **one training at a time** while the other apps stay open: seed 142 continues alone; seed 242 resumes
  exactly from 1,003,520 when 142 finishes.
- New: `scripts/memory_guard.py` checks Windows' available memory every 30 s and, if it is under 3 GB twice in a
  row, stops one of my jobs (analysis jobs first, then the newest training). Log: `logs/memory_guard.log`.
  New config values: `MIN_FREE_PC_MEMORY_GB = 3.0`, `NICE_LEVEL = 10`; all my jobs now run at nice 10.

Mid-run observations (not judged yet, nothing changed; the final copies are what count):
seed 142's 20-track checks stayed at 246-317 up to 1M decisions (covers only ~40% of the track, almost never
leaves the playing area); seed 242 peaked at 747 (504k) and was at 539 at 1M.

## Step 3: Color test (plan written 2026-10-17 23:20, before any Agent B result)
Nothing in this plan needs a new decision from the director: the recipe was decided for A and B alike
(`config.RECIPE = "entropy"`), the color rules were fixed in Step 1, and the time was announced in Step 2.

**What runs**
1. **A on unseen colors.** A's three final copies on the 100 analysis tracks (20000-20099) with unseen colors:
   color seed = track number. B trains only on color seeds 1,000,000-1,999,999, so no agent ever saw these.
   The normal-color results on the same tracks come from Step 2.
2. **Train B:** seeds 42, 142, 242; same recipe, same 4M decisions, final copies. The only difference from A:
   a new random color scheme every training race (Gymnasium's color formula, own random stream, so the
   track for a given seed is unchanged).
3. **Evaluate B** the same way as A: same 100 analysis tracks, normal and unseen colors (identical color seeds).
   Every race's actions are saved; replays are checked against the recorded results.
4. **Driving checks** (`scripts.inspect_driving`) for B on both color conditions and for A on unseen colors.

**Test plan (fixed now)** - `scripts/compare_agents.py`
- Unit = a track. For each agent and color condition, a track's score = the mean over that agent's 3 seeds.
  Pairs = same track, same colors.
- Four comparisons (first minus second):
  1. A unseen - A normal (does A suffer on new colors?)
  2. **B unseen - A unseen (the main question)**
  3. B normal - A normal (the price of robustness: negative = B pays a price on normal colors)
  4. B unseen - B normal
- Wilcoxon signed-rank test, two-sided, on the 100 per-track differences; Holm correction across the four;
  a difference is called significant only if its Holm-adjusted p < 0.05. Also reported: mean and median
  difference, 95% paired-bootstrap interval (10,000 resamples, seed 0), tracks better / worse / tied.
- Every metric for every agent x color condition, per seed and as the mean of seeds: score, solved (900),
  track covered, laps (strict), laps (game's rule), off track.
- Limits, stated in advance: the tests compare these six trained drivers across tracks. With 3 training runs per
  agent they do not measure how much a fresh training run would vary, so each seed's numbers are shown and any
  disagreement between seeds is reported. "Proven" is never used without a significant paired test.
- The 800 pass mark is not changed. The hidden test tracks (10000-10099) are not touched.
- The script was checked on made-up data with known effects (-200, +100, 0, -100): it recovered all four and
  called the zero effect not significant.

**Timing.** With the director's other apps open, memory allows one training at a time, about 4.5-5 h per B seed,
so about 14-15 h for all three, plus about 1.5 h of evaluations. B starts once A seed 242 is done. A second
training only starts when Windows has at least 5.5 GB available (the 3 GB rule plus about 2.5 GB for the run);
the memory guard enforces the 3 GB rule either way.

## Slow driving: diagnosis (no training; plan for a fix follows once the diagnosis is done)
The director asked for a plan to fix the slow driving, with the judging rule written in advance, and **no
training without the director's go**. First, cheap checks on seed 42's final copy (`entropy`), validation tracks only.

**1. Speed diagnosis** (`scripts/diagnose_speed.py`, `results/step2/speed_diagnosis/`; all 100 races replayed exactly)
- The car drives at **one steady cruising speed everywhere**: gas about 0.09 whatever the road ahead does;
  it never uses more than half gas (highest 0.51), never full gas, and almost never brakes.
- It is **fastest right before sharp turns** (average speed 57 vs 49 on straights), because speed builds up over
  each straight with steady gas. In front of sharp turns all four wheels are off the road 17% of the time
  (straights: 4%), and brakes only 3.4% of that time.
- Its average speed (51.9) is almost exactly what a lap needs to finish in 20 seconds (51.7): 78 of 100 races
  ran out of time, 55 of them with 90%+ of the track covered.

**2. Gas-scaling what-if** (`scripts/diagnose_gas_scaling.py`, rule written before running): replay the same
driver on the 100 validation tracks with its gas multiplied by 1.5, 2 and 3 (steering and brake untouched).
- Check: factor 1.0 must reproduce the normal evaluation exactly (781.8), otherwise the test is broken.
- **"Steering copes with more speed"** if some factor >= 1.5 gives a higher mean score than 1.0 **and** at most
  10 percentage points more off-track races. Then the limit is the throttle the AI learned.
- **"Steering can't cope"** if 1.5x already adds more than 10 points of off-track races or loses more than 50
  points of mean score. Then the AI would need to learn braking and anticipation, not just more gas.
- Anything else: "mixed", reported as such.
