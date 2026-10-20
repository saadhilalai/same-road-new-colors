# Retraining plan with recipe 'beta': gate decisions

Rule: PROGRESS.md, written before training. Validation tracks, normal colors, final copies.

## a42 (B recipe: beta), 2026-10-19 22:10

- A seed 42 (beta): mean 778.9 +/- 177.3, laps 8.0%, off track 3.0%, pass mark 800: not reached
- Rule: A seed 42 (recipe or code broken if not) needs a mean of at least 500: 778.9 -> PASSED

## a42 (B recipe: beta), 2026-10-19 22:12

- A seed 42 (beta): mean 778.9 +/- 177.3, laps 8.0%, off track 3.0%, pass mark 800: not reached
- Rule: A seed 42 (recipe or code broken if not) needs a mean of at least 500: 778.9 -> PASSED

## b42 (B recipe: beta), 2026-10-20 03:45

- B seed 42 (beta): mean 728.1 +/- 243.5, laps 7.0%, off track 9.0%, pass mark 800: not reached
- Rule: B seed 42 'can drive' (beta) needs a mean of at least 100: 728.1 -> PASSED

## two_seeds (B recipe: beta), 2026-10-20 15:05

- A seed 42 (beta): mean 778.9 +/- 177.3, laps 8.0%, off track 3.0%, pass mark 800: not reached
- A seed 142 (beta): mean 806.7 +/- 162.2, laps 9.0%, off track 1.0%, pass mark 800: reached
- B seed 42 (beta): mean 728.1 +/- 243.5, laps 7.0%, off track 9.0%, pass mark 800: not reached
- B seed 142 (beta): mean 719.5 +/- 175.4, laps 0.0%, off track 0.0%, pass mark 800: not reached
- A: two-seed mean 792.8 vs old recipe 'entropy' 475.7 (allowed: at most 50 lower, so at least 425.7) -> PASSED
- B: two-seed mean 723.8 (needs at least 300) -> PASSED
- B vs old recipe 'entropy' (two-seed mean -12.2): mean difference +736.0, better on 100 of 100 tracks, worse on 0; paired Wilcoxon (two-sided) p = 3.9e-18 (needs p < 0.05 and a positive difference) -> PASSED
- Two-seed rule: PASSED, the new recipe is adopted
