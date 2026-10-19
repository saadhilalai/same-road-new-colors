# Step 3 color test (analysis tracks, final copies, recipe 'entropy')

## Results per seed (and mean of seeds)

| agent   | colors   | seed          |   races |   mean_score |   std_score | solved_900   |   mean_track_covered_pct |   lap_completed_pct |   lap_finished_gym_pct |   off_track_pct |
|:--------|:---------|:--------------|--------:|-------------:|------------:|:-------------|-------------------------:|--------------------:|-----------------------:|----------------:|
| A       | normal   | 42            |     100 |        780.7 |       178.4 | False        |                     88.3 |                12   |                   12   |             4   |
| A       | normal   | 142           |     100 |        152.3 |       116.1 | False        |                     25.2 |                 0   |                    0   |             0   |
| A       | normal   | 242           |     100 |        642   |       161.3 | False        |                     74.2 |                 0   |                    0   |             0   |
| A       | normal   | mean of seeds |     300 |        525   |       151.9 | False        |                     62.6 |                 4   |                    4   |             1.3 |
| A       | unseen   | 42            |     100 |         89.7 |       288.2 | False        |                     22.6 |                 1   |                    1   |            64   |
| A       | unseen   | 142           |     100 |        -87.1 |        19.2 | False        |                      1.4 |                 0   |                    0   |             4   |
| A       | unseen   | 242           |     100 |          1.5 |       129.1 | False        |                     10.2 |                 0   |                    0   |             1   |
| A       | unseen   | mean of seeds |     300 |          1.4 |       145.5 | False        |                     11.4 |                 0.3 |                    0.3 |            23   |
| B       | normal   | 42            |     100 |        -22   |        23.7 | False        |                      7.9 |                 0   |                    0   |             1   |
| B       | normal   | 142           |     100 |          2   |        32.9 | False        |                     10.2 |                 0   |                    0   |             0   |
| B       | normal   | 242           |     100 |         78   |        63.7 | False        |                     17.8 |                 0   |                    0   |             0   |
| B       | normal   | mean of seeds |     300 |         19.3 |        40.1 | False        |                     12   |                 0   |                    0   |             0.3 |
| B       | unseen   | 42            |     100 |        -10.5 |        42.7 | False        |                      9   |                 0   |                    0   |             1   |
| B       | unseen   | 142           |     100 |         25.7 |        77.4 | False        |                     12.7 |                 0   |                    0   |             1   |
| B       | unseen   | 242           |     100 |         79.5 |        89.1 | False        |                     17.9 |                 0   |                    0   |             0   |
| B       | unseen   | mean of seeds |     300 |         31.6 |        69.7 | False        |                     13.2 |                 0   |                    0   |             0.7 |

## Paired tests (unit = track; score = mean over each agent's seeds)

- **A: unseen minus normal colors (does A suffer on new colors?)**: mean difference -523.6 (95% CI -558.9 to -487.3), median -578.0; better on 1 tracks, worse on 99, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 4.14e-18, Holm-adjusted p = 1.56e-17 -> **significant (Holm p < 0.05)**
- **Unseen colors: B minus A (main question)**: mean difference +30.2 (95% CI +3.9 to +54.4), median +72.7; better on 76 tracks, worse on 24, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 0.000247, Holm-adjusted p = 0.000494 -> **significant (Holm p < 0.05)**
- **Normal colors: B minus A (price of robustness if negative)**: mean difference -505.7 (95% CI -528.7 to -482.2), median -535.4; better on 0 tracks, worse on 100, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 3.9e-18, Holm-adjusted p = 1.56e-17 -> **significant (Holm p < 0.05)**
- **B: unseen minus normal colors**: mean difference +12.2 (95% CI +3.5 to +21.1), median +9.8; better on 61 tracks, worse on 39, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 0.00962, Holm-adjusted p = 0.00962 -> **significant (Holm p < 0.05)**

These tests compare the trained drivers across tracks. With 3 training runs per agent, they do not measure how much a new training run of the same recipe would vary.

