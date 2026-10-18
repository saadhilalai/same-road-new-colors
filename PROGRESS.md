# Progress log

Race-car AI that learns to drive from pixels only (Gymnasium CarRacing-v3, PPO),
testing whether domain randomization (random colors in training) makes it robust to unseen colors.

## Morning summary (night of 2026-10-17/18; written 05:30, updated 07:25)

**Finished overnight**
- **Step 2 is done** (details in "Step 2 results" below). Agent A, recipe `entropy`, final copies, normal colors:

  | Agent A | Validation (100 tracks) | Analysis (100 tracks) | Laps (strict), analysis | Off track, analysis |
  |---|---|---|---|---|
  | Seed 42 | 781.8 | 780.7 | 12% | 4% |
  | Seed 142 | 169.6 | 152.3 | 0% | 0% |
  | Seed 242 | 644.1 | 642.0 | 0% | 0% |
  | Mean of seeds | 531.8 | 525.0 | 4% | 1.3% |

  **All three miss the 800 pass mark**; none is solved (900). The seeds differ a lot: seed 142 creeps along at a
  quarter of seed 42's speed. All three drive at one steady speed and never slow down for corners. No point-farming tricks.
- **Step 3, first part done:** Agent A on unseen colors (analysis tracks) collapses for every seed:
  42: 780.7 -> 89.7, 142: 152.3 -> -87.1, 242: 642.0 -> 1.5 (mean 525.0 -> 1.4). Seed 42 drives off the map at the
  first corner in 64% of races; seeds 142 and 242 barely move. (No statistical test yet: the written plan compares A
  with B, which needs B's results.)
- **Step 3 plan and test rules written down before any B result** (Wilcoxon per track, Holm over 4 tests).
- **Slow-driving diagnosis** (no training): the car cruises at the speed a lap barely needs, faster before sharp
  turns than on straights, and almost never brakes. Giving it more gas makes it crash more (1.5x gas: -101 points,
  +12 points of off-track races). A **plan with two candidate fixes and a judging rule fixed in advance** is in
  "Plan: fixing the slow driving". Not started.

**Running now**
- **Agent B seed 42** (random colors, recipe `entropy`) **is not learning to drive.** On the 20 normal-color check
  tracks it scores -40 to -10, covering about 8% of the track; at the same point A42 was at 720-820. Its training races (random
  colors) rose to a mean of about 120 by 500k decisions, then slid to -5 by 1.75M. Probably not a code bug: it
  drove 900+ on a few early races. Details: "Agent B training (2026-10-18)". Nothing has been changed; the plan
  runs as written.
- **07:06: the memory guard stopped B seed 42** (Windows 2.6 GB available; nothing lost but 1 minute). New rule,
  written before use: a job restarts once Windows has 5.5 GB available for 10 minutes. A queue
  (`scripts/job_queue.sh`) now runs everything one at a time: B42 to the end (about 2.2 h of training left),
  B42's evaluation, then B142 and B242 (about 4.5 h each), their evaluations, and the A-vs-B tests.
- One job at a time: with the PC's other apps open, Windows had only 2.3-6 GB of memory available. Your 3 GB rule
  stopped my work three times: I stopped seed 242's training by hand at 23:15 (resumed exactly at 02:00), and the memory
  guard (`scripts/memory_guard.py`) stopped an evaluation at 05:07 and B42 at 07:06. All my jobs run at low CPU
  priority (nice 10).

**What I need from you**
1. **The slow-driving fix (D1-D3 in the plan):** run the fix trial or not, which candidates, which seeds. Note: if a
   fix is adopted, A and B both have to be retrained with it, and the B runs going on now become a "before" result.
   If you want the trial first, I can pause B (it resumes exactly later).
2. **Agent B (new):** B42 may finish far below A. If all B seeds fail like this, the main question
   ("does B handle unseen colors better?") gets a weak answer: B would be worse than A on normal colors and both would fail on unseen ones.
   Options: **(a) carry on as written** (running now; costs only PC time; B142/B242 may do better, as A's seeds
   varied from 152 to 781); **(b) after B42, pause B** and first fix the unstable training updates for A and B
   together (this could share one trial with the slow-driving fix); **(c) train B longer**, but then A and B
   differ in two ways unless A is retrained as long too. **My recommendation: (a) for now**, and decide on (b)
   together with item 1, since a fix that is adopted means retraining A and B anyway.
3. **Optional:** freeing about 2 GB of memory on Windows would let two jobs run at once (about twice as fast).
4. **Reminder:** pause Windows updates while runs go overnight (it restarted the PC at 01:59 the night before).

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

### Agent B training (2026-10-18)
**07:06: the memory guard stopped B seed 42** at about 1.77M decisions: Windows had only 2.6 GB available.
The guard read 5.5-6 GB before 06:40, then 4.8 GB, 3.7 GB and 2.6 GB with the training running. The other apps
used more memory. Without the training, Windows had 4.6 GB available.
The last checkpoint (1,753,088, full state saved) is complete, so the stop costs about 16k decisions (1 minute).

**Restart rule (written 07:20, before first use).** After a memory stop, a job restarts only once Windows has had
at least **5.5 GB available for 10 minutes without a dip**. That is the 3 GB floor plus about 2 GB per job, plus a
margin, so the restart doesn't trip the guard again at once. The values are in `config.py`
(`RESUME_MIN_FREE_PC_MEMORY_GB`, `RESUME_STEADY_MINUTES`); `scripts/wait_for_memory.py` does the waiting.
`scripts/job_queue.sh` runs the jobs one at a time and logs to `logs/job_queue.log`. It waits for memory before each
start. After a memory stop, a training resumes exactly and an evaluation reruns (its results are deterministic).
Any other failure ends the queue.
Order: B42 training -> B42 evaluation (analysis tracks, normal + unseen colors) -> B42 driving checks -> B142
training -> B242 training -> their evaluations and driving checks -> the Step 3 tests. B42 is now evaluated right
after its own training, not after all three trainings, so a finished B can be looked at early. No test, track set
or rule changes.

**Mid-run observation (not judged, nothing changed; the final copy is what counts): B seed 42 learned a little,
then got worse.** Training races (B's are on random colors, A's on normal colors), seed 42 in both:

| Decisions | A: races | A: mean score | A: % races >= 900 | B: races | B: mean score | B: % races >= 900 |
|---|---|---|---|---|---|---|
| 0-250k | 525 | 424.0 | 5.5 | 512 | 16.0 | 0.2 |
| 250-500k | 516 | 809.5 | 23.3 | 518 | 123.4 | 0.4 |
| 500-750k | 533 | 751.8 | 25.5 | 518 | 123.6 | 0 |
| 750k-1M | 525 | 770.4 | 26.5 | 512 | 55.0 | 0 |
| 1-1.25M | 522 | 749.2 | 21.1 | 535 | 21.3 | 0 |
| 1.25-1.5M | 510 | 765.1 | 14.3 | 520 | 13.7 | 0 |
| 1.5-1.75M | 525 | 740.4 | 14.7 | 521 | -4.9 | 0 |

- **On the 20 normal-color check tracks, B42 never got going:** -40 (250k), -12, -10, -20, -10, -12, -21 (1.75M), covering
  8-9% of the track. A42 was at 822 by 250k.
- **Probably not a code bug.** B drove 900+ on a few early training races (best 905.6), so random colors render and
  the scoring works. The Step 1 environment checks still hold.
- **Colors matter.** In the second half of B's training races so far, races with the least contrast between road and background (lowest
  quarter) averaged -24; the highest quarter averaged +43. Normal colors are in the lowest-contrast quarter by
  this measure (gray road vs green background differ in one channel only), so normal colors are among the harder
  schemes for B.
- **Training health:** A and B lose exploration noise at the same pace (std 0.12 -> 0.03 by 1.75M). Both change
  their driving a lot at every update (approx_kl 0.3-0.9, clip fraction 0.57-0.75; typical PPO values are
  roughly 0.01-0.05 and 0.1-0.2). **Likely reason (untested):** these unstable updates, the same weakness that sank the
  Zoo pilot, matter more on the harder random-color task, and the shrinking noise leaves B little room to recover.
- **Default (running):** carry on with the written plan (all three B seeds, final copies). A's seeds differed hugely
  (152 to 781), so B142/B242 may not follow B42. The question for the director is in the morning summary.

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

**Gas-scaling result (02:15): "steering can't cope"** - by the rule written above. `results/step2/gas_scaling/summary.md`

| Gas x | Mean score | Covered | Laps | Off track | Out of time | ...with 90%+ covered |
|---|---|---|---|---|---|---|
| 1.0 (exact match with the normal evaluation) | 781.8 | 88.5% | 16% | 6% | 78% | 55 |
| 1.5 | 680.5 | 79.4% | 4% | 18% | 78% | 36 |
| 2.0 | 526.5 | 65.1% | 2% | 36% | 62% | 11 |
| 3.0 | 368.9 | 49.0% | 0% | 33% | 67% | 6 |

At 1.5x gas the driver already loses 101 points and leaves the playing area 12 points more often. So the
slowness is not just a timid throttle on top of good steering: at higher speed this driver misses corners.
A faster driver would have to *learn* to brake before corners and speed up on straights, which it never does now.

**All three seeds are slow** (training races, from 1.5M decisions on): seed 42 covers about 85% of the track per race,
seed 242 50-67%, and seed 142 only 22-25%. Seeds 142 and 242 never leave the playing area and every race lasts the
full 1,000 frames: they creep along the road. Seed 142's final copy: 169.6 on the 100 validation tracks, 152.3 on the
100 analysis tracks.

## Plan: fixing the slow driving (written 2026-10-18 02:35; NOT started, needs the director's go)

**What we know** (sections above): all three seeds settle on driving slowly and steadily. Seed 42 cruises at
almost exactly the speed a lap needs and runs out of time; seeds 142 and 242 creep (seed 142: average speed 13
vs 52 for seed 42). None of them slows down for corners or speeds up on straights. Giving seed 42 more gas makes
it crash more (gas-scaling test), so the fix has to teach braking and anticipation, not just "more gas".

**Likely reasons (untested)**
1. *Exploration of the pedals dies early.* The AI's random "trying things" shrinks from 0.135 to about 0.01.
   Gas and brake are two separate outputs, each cut off at 0, so a driver that has settled on "no brake" almost
   never gets to try braking (seed 42 brakes in 0.7% of decisions) and never learns its value before corners.
2. *Running out of time doesn't hurt it in training.* At the 1,000-frame limit, Stable-Baselines3 treats the race
   as if it went on (it "bootstraps" the value). So in training, unfinished track at time-out costs nothing,
   while a crash costs -100 plus everything after it. Slow and safe looks good to the AI. Also, the AI
   can't see how much time is left: the screen has no clock.

**Two candidate fixes, each changing exactly one thing**
- **F1 "one pedal"**: gas and brake become one output from -1 to 1 (above 0 = gas, below 0 = brake). A thin
  wrapper turns it back into the game's 3 controls; game, rewards and score are unchanged. A neutral driver then
  tries braking as often as accelerating. Targets reason 1.
- **F2 "time counts"**: the 1,000-frame limit becomes the real end of a race in training (no bootstrapping), and
  the AI gets a time-left bar drawn into the black dashboard strip of its image (otherwise it couldn't know when
  the end comes). Targets reason 2. Changes what the AI sees, so it is a bigger design change than F1.

**Test design**
- Each candidate trains on seeds 42 **and** 142 (the current recipe's best and worst seed; one seed alone misled
  us before), 4M decisions, final copies. Everything else is the current recipe (`entropy`).
- Judged only on the 100 validation tracks, normal colors. Test tracks stay hidden.
- Cost: 4 runs of about 4.5-5 h. With the PC's other apps open, memory allows one at a time (about 19 h); two at
  a time (about 10 h) if Windows has about 2 GB more free.

**Judging rule (fixed now, before any training)** - a candidate "works" only if all of these hold:
1. **Better scores:** per track, the mean of its two seeds beats the current recipe's mean of the same two seeds
   (currently 781.8 and 169.6 on validation, mean 475.7). Paired Wilcoxon test over the 100 tracks, two-sided,
   Holm-corrected over the 2 candidates: mean difference > 0 and adjusted p < 0.05.
2. **Mechanism:** for both seeds, the speed diagnosis shows it is slower before sharp turns than on straights
   (now it is faster), and fewer races run out of time with 90%+ covered than with the current recipe.
3. **Safety:** off-track races at most 10 percentage points above the current recipe (same seeds).
4. **No tricks** in the driving check (driving backwards, shortcuts, point farming).
5. The 800 pass mark is reported per seed and not changed.
- If both work: the higher mean wins (the paired test between them is reported). If neither works: keep the
  current recipe and report it.

**What it would mean for the project:** A and B must use the same recipe. If a fix wins, both agents are retrained
with it (the trial's seed 42 and 142 runs count for A; then A seed 242 and B seeds 42, 142, 242: 4 more runs).
The B runs started tonight with the current recipe would then be a "before" comparison, not the final result.

**Decisions needed from the director**
- D1: run the fix trial, or keep the current recipe and report the slowness as a limitation? (The A-vs-B
  colour question doesn't need a fast driver, but seed 142's crawl drags Agent A's average down.)
- D2: if yes, which candidates (F1, F2 or both) and which seeds (42 + 142 recommended; 42 alone saves half the time).
- D3: allow two trainings at once by freeing about 2 GB of memory on Windows (halves the time)?

## Step 2 results: Agent A (finished 2026-10-18 05:25)
Agent A = recipe `entropy`, seeds 42 / 142 / 242, 4M decisions each, **final copies**, normal colors.
Every evaluation race was replayed exactly (100/100 per seed and track set). Tables:
`results/step2/agentA_validation_normal.md`, `results/step2/agentA_analysis_normal.md`.
Learning curves: `results/step2/learning_curves.png`.

| Agent A | Validation (100) | Analysis (100) | Analysis: covered | Laps (strict) | Laps (game's rule) | Off track |
|---|---|---|---|---|---|---|
| Seed 42 | 781.8 | 780.7 +/- 178.4 | 88.3% | 12% | 12% | 4% |
| Seed 142 | 169.6 | 152.3 +/- 116.1 | 25.2% | 0% | 0% | 0% |
| Seed 242 | 644.1 | 642.0 +/- 161.3 | 74.2% | 0% | 0% | 0% |
| **Mean of seeds** | **531.8** | **525.0** | 62.6% | 4% | 4% | 1.3% |

- **Not solved** (900) by any seed. **The 800 pass mark is missed by all three seeds** (best: seed 42, 780.7-781.8).
- **Large differences between seeds** (152 to 781 on the same tracks). Picking the recipe on seed 42 alone made it
  look better than it is: seed 42 turns out to be the best of the three.
- Learning curves: seed 42 learned within 250k decisions and stayed at 750-820 on its 20-track checks. Seed 142
  never got past about 320 and settled at 120-170. Seed 242 reached 747 at 504k, slid to 417 at 1.25M (like the
  Zoo pilot's collapse), then recovered to 696 by the end.

**How the three seeds drive** (frame sheets, path maps and driving numbers on the analysis tracks;
`results/driving/A_seed*_entropy_analysis_normal/`)
- **All three:** smooth, no wobble (0.16-0.66 steering reversals per second), almost no braking (0.0-0.7% of
  decisions), never full gas, and **one steady speed** whatever the road does. None of them slows down for corners.
- **Seed 42 (average speed 51):** centred, slightly cuts the inside of curves, follows almost every bend. Runs out
  of time with 90%+ of the track done in most races. Its typical failure: overshoots a sharp hairpin onto the grass,
  then drives straight until it leaves the map (4% of races), or, once lost, rejoins the road the wrong way.
- **Seed 142 (average speed 13):** creeps. When it misses a bend at that low speed it rolls onto the grass and
  stops there (20% of its time with all wheels on grass). Never leaves the playing area, covers a quarter of the track.
- **Seed 242 (average speed 36):** very clean (all wheels on grass only 1.9% of the time, never leaves the map) but
  slow: covers about three quarters of the track in the time. At some hairpins it slows almost to a stop and gets stuck.
- **No point-farming tricks.** Driving backwards: at most 0.5% of the time on average. The shortcut counter flagged
  21 / 17 / 1 races (seeds 42 / 142 / 242). Those are mostly bad races with lots of grass time; the two I replayed and
  drew were a car that lost the road and looped back over road it had already driven (no points for that), and a
  car that cut the inside of one bend and then stopped on the grass. Not a way of collecting extra points.

**Notes**
- The recipe's main weakness is the slow driving (diagnosis and a fix plan above; needs the director's go).
- The 05:07 memory-guard stop: when Agent B seed 42 started training and seed 242's evaluation started next to it,
  Windows' available memory fell to 2.3 GB and the guard stopped the evaluation. With the PC's other apps open,
  only one of my jobs fits at a time, so B was stopped (2 minutes in, resumed exactly from checkpoint 0 at 05:23)
  and the Step 2 evaluations ran first.
