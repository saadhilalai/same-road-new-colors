# Final test plan (Step 6)

Written and committed to git **before** the hidden test tracks were used for anything. `scripts/final_test.py`
refuses to start unless this file and the script itself are committed unchanged, and refuses to run a second time.

## Agents
- **Agent A**: trained on normal colors only. **Agent B**: trained with a new random color scheme every race.
- Both: recipe "beta" (PPO with a Beta policy and one pedal, 5M decisions), training seeds 42, 142, 242,
  **final copies** (`model_final.zip`). Nothing was picked by score.
- **Baseline, untrained car**: A seed 42 before any training (checkpoint 0). An untrained AI's best action is to stand
  still, so it is shown as it really starts training: picking random actions from its starting policy (random seed 0).
  Normal colors only. It is a floor for the table and the source of the "untrained car" video, not part of any test.

## Races
- **Tracks**: the 100 hidden test tracks, seeds 10000-10099, never used before (only listed in `config.py`).
- **Colors**: normal; and unseen, where the race on track T uses color seed T (10000-10099). Agent B trained only on
  color seeds 1,000,000-1,999,999, so no agent has ever seen these color schemes. They come from the same color
  formula as B's training colors.
- Every race is run once, with the AI's best action (no randomness, except the untrained baseline), and every action is
  recorded so any race can be replayed exactly. Videos (Step 7) replay these exact races and are checked against
  the recorded results.

## Metrics (the project's definitions)
Score (official reward), solved (mean score 900+), track covered (% of tiles), lap completed (every tile before time
ran out), lap finished by the game's own rule, off track (left the playing area). Reported per seed and as the mean of
seeds. The 800 pass mark is reported per seed, unchanged.

## Statistical tests (the same four as Step 3)
- Unit = a track. For each agent and color condition, a track's score = the mean over that agent's 3 seeds.
  Pairs = same track, same colors.
- Comparisons (first minus second):
  1. A unseen - A normal (does A suffer on new colors?)
  2. **B unseen - A unseen (the main question)**
  3. B normal - A normal (the price of robustness; negative = B pays a price)
  4. B unseen - B normal
- Wilcoxon signed-rank test, two-sided, on the 100 per-track differences; Holm correction over the four.
  **Significant = Holm-adjusted p < 0.05.** Also reported: mean and median difference, 95% paired-bootstrap interval
  (10,000 resamples, seed 0), tracks better / worse / tied.
- Plain-English verdicts, fixed now:
  - Main question: test 2 significant and positive -> "B handles unseen colors better than A on the hidden test
    tracks". Not significant -> "no difference shown". Significant and negative -> "A handles them better".
  - Price of robustness: test 3 significant and negative -> "B pays a price on normal colors" (with its size);
    otherwise "no price shown".
- Also shown, without a test: each seed's numbers, and whether every B seed beats every A seed on unseen colors.

## Limits, stated in advance
- The tests compare these six trained drivers across 100 tracks. With 3 training runs per agent they do not measure
  how much a fresh training run would vary.
- "Unseen colors" means new color schemes from the same random formula B trained on. Other kinds of change (lighting,
  textures, a different color formula) are not tested.
- The game ends a race early only when every road tile has been touched (found in Step 3), so most races run to the
  time limit. That is the same for every agent.

## Running it once
- `python -m scripts.final_test` evaluates every agent, seed and color set on the test tracks, then runs
  `scripts.compare_agents --tracks test --out-dir results/final_test` (the Step 3 analysis, unchanged).
- If a memory stop interrupts it, the restart only finishes the missing runs (same races, same AI, same results), and
  no result is looked at before all runs are done. No re-running and no changes after seeing results.

## Videos (Step 7): which races, fixed now
Every clip replays a race of this test exactly (checked against its recorded result), full size, 50 frames per second.
1. **Untrained car**: the untrained baseline on the first test track (10000), normal colors.
2. **Agent A driving well on normal colors**: the A seed with the best normal-color mean, its best race (labeled so).
3. **A failing vs B on unseen colors, side by side**: seed 42, on the test track whose B-minus-A gap (seed 42, unseen
   colors) is the median of the 100 tracks, so it is a typical track, not the most flattering one.
4. **Split screen** (what a human sees | the AI's real 64x64 view | the blur-test heat map): B seed 42, same race as 3.
5. **Five B-roll clips** (10 seconds each, from 1 s in, no captions): B's three best unseen-color races and A's and B's
   best normal-color races, on five different tracks.
