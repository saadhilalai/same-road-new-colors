# Step 3 color test (analysis tracks, final copies, recipes: A 'beta', B 'beta')

## Results per seed (and mean of seeds)

| agent   | colors   | seed          |   races |   mean_score |   std_score | solved_900   |   mean_track_covered_pct |   lap_completed_pct |   lap_finished_gym_pct |   off_track_pct |
|:--------|:---------|:--------------|--------:|-------------:|------------:|:-------------|-------------------------:|--------------------:|-----------------------:|----------------:|
| A       | normal   | 42            |     100 |        783.9 |       180.4 | False        |                     88.6 |                10   |                   10   |             5   |
| A       | normal   | 142           |     100 |        793.6 |       183.9 | False        |                     89.3 |                12   |                   12   |             3   |
| A       | normal   | 242           |     100 |        819   |       151.7 | False        |                     91.9 |                19   |                   19   |             6   |
| A       | normal   | mean of seeds |     300 |        798.8 |       172   | False        |                     89.9 |                13.7 |                   13.7 |             4.7 |
| A       | unseen   | 42            |     100 |         45.6 |       261   | False        |                     15.3 |                 0   |                    0   |            14   |
| A       | unseen   | 142           |     100 |        113.2 |       275   | False        |                     22.5 |                 1   |                    1   |            21   |
| A       | unseen   | 242           |     100 |         -5.9 |       137.9 | False        |                     10.4 |                 0   |                    0   |            32   |
| A       | unseen   | mean of seeds |     300 |         51   |       224.6 | False        |                     16.1 |                 0.3 |                    0.3 |            22.3 |
| B       | normal   | 42            |     100 |        720.5 |       241.2 | False        |                     83   |                11   |                   11   |            13   |
| B       | normal   | 142           |     100 |        715.7 |       160.7 | False        |                     81.6 |                 0   |                    0   |             0   |
| B       | normal   | 242           |     100 |        835.5 |       132.7 | False        |                     93   |                30   |                   30   |             1   |
| B       | normal   | mean of seeds |     300 |        757.2 |       178.2 | False        |                     85.9 |                13.7 |                   13.7 |             4.7 |
| B       | unseen   | 42            |     100 |        544.2 |       330.3 | False        |                     64.8 |                 6   |                    6   |             8   |
| B       | unseen   | 142           |     100 |        569.5 |       300.3 | False        |                     68.4 |                 5   |                    5   |            22   |
| B       | unseen   | 242           |     100 |        478.1 |       321.8 | False        |                     60.1 |                 3   |                    3   |            36   |
| B       | unseen   | mean of seeds |     300 |        530.6 |       317.5 | False        |                     64.4 |                 4.7 |                    4.7 |            22   |

## Paired tests (unit = track; score = mean over each agent's seeds)

- **A: unseen minus normal colors (does A suffer on new colors?)**: mean difference -747.9 (95% CI -787.5 to -705.4), median -808.5; better on 0 tracks, worse on 100, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 3.9e-18, Holm-adjusted p = 1.56e-17 -> **significant (Holm p < 0.05)**
- **Unseen colors: B minus A (main question)**: mean difference +479.7 (95% CI +425.3 to +535.0), median +491.5; better on 96 tracks, worse on 4, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 1.41e-17, Holm-adjusted p = 4.24e-17 -> **significant (Holm p < 0.05)**
- **Normal colors: B minus A (price of robustness if negative)**: mean difference -41.6 (95% CI -69.7 to -12.6), median -35.6; better on 34 tracks, worse on 66, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 0.00405, Holm-adjusted p = 0.00405 -> **significant (Holm p < 0.05)**
- **B: unseen minus normal colors**: mean difference -226.6 (95% CI -274.3 to -180.5), median -210.0; better on 18 tracks, worse on 82, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 1.1e-12, Holm-adjusted p = 2.2e-12 -> **significant (Holm p < 0.05)**

These tests compare the trained drivers across tracks. With 3 training runs per agent, they do not measure how much a new training run of the same recipe would vary.

