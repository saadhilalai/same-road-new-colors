# Final test (test tracks, final copies, recipes: A 'beta', B 'beta')

## Results per seed (and mean of seeds)

| agent   | colors   | seed          |   races |   mean_score |   std_score | solved_900   |   mean_track_covered_pct |   lap_completed_pct |   lap_finished_gym_pct |   off_track_pct |
|:--------|:---------|:--------------|--------:|-------------:|------------:|:-------------|-------------------------:|--------------------:|-----------------------:|----------------:|
| A       | normal   | 42            |     100 |        801.1 |       155.8 | False        |                     90.1 |                11   |                   11   |             4   |
| A       | normal   | 142           |     100 |        796.2 |       202   | False        |                     89.5 |                19   |                   19   |             4   |
| A       | normal   | 242           |     100 |        824   |       135.6 | False        |                     92.2 |                22   |                   22   |             4   |
| A       | normal   | mean of seeds |     300 |        807.1 |       164.5 | False        |                     90.6 |                17.3 |                   17.3 |             4   |
| A       | unseen   | 42            |     100 |         69.2 |       278.7 | False        |                     17.7 |                 0   |                    0   |            16   |
| A       | unseen   | 142           |     100 |        172.8 |       329.1 | False        |                     28.1 |                 3   |                    3   |            16   |
| A       | unseen   | 242           |     100 |         33.9 |       193.1 | False        |                     14.4 |                 0   |                    0   |            27   |
| A       | unseen   | mean of seeds |     300 |         92   |       267   | False        |                     20.1 |                 1   |                    1   |            19.7 |
| B       | normal   | 42            |     100 |        721.7 |       223.5 | False        |                     82.4 |                 6   |                    6   |             5   |
| B       | normal   | 142           |     100 |        688.2 |       179.5 | False        |                     78.9 |                 0   |                    0   |             1   |
| B       | normal   | 242           |     100 |        807.9 |       176.7 | False        |                     90.7 |                18   |                   18   |             5   |
| B       | normal   | mean of seeds |     300 |        739.3 |       193.2 | False        |                     84   |                 8   |                    8   |             3.7 |
| B       | unseen   | 42            |     100 |        622.5 |       290.4 | False        |                     72.4 |                10   |                   10   |             4   |
| B       | unseen   | 142           |     100 |        618.5 |       277.9 | False        |                     72.9 |                 7   |                    7   |            17   |
| B       | unseen   | 242           |     100 |        503.3 |       310.1 | False        |                     62.1 |                 4   |                    4   |            30   |
| B       | unseen   | mean of seeds |     300 |        581.4 |       292.8 | False        |                     69.1 |                 7   |                    7   |            17   |

## Paired tests (unit = track; score = mean over each agent's seeds)

- **A: unseen minus normal colors (does A suffer on new colors?)**: mean difference -715.1 (95% CI -758.3 to -669.2), median -789.5; better on 1 tracks, worse on 99, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 4.14e-18, Holm-adjusted p = 1.66e-17 -> **significant (Holm p < 0.05)**
- **Unseen colors: B minus A (main question)**: mean difference +489.5 (95% CI +430.8 to +545.0), median +538.0; better on 93 tracks, worse on 7, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 5.82e-17, Holm-adjusted p = 1.75e-16 -> **significant (Holm p < 0.05)**
- **Normal colors: B minus A (price of robustness if negative)**: mean difference -67.8 (95% CI -100.3 to -36.8), median -57.5; better on 37 tracks, worse on 63, tied 0 (100 tracks, seeds 3 vs 3); Wilcoxon p = 0.000228, Holm-adjusted p = 0.000228 -> **significant (Holm p < 0.05)**
- **B: unseen minus normal colors**: mean difference -157.8 (95% CI -195.9 to -119.6), median -118.8; better on 20 tracks, worse on 79, tied 1 (100 tracks, seeds 3 vs 3); Wilcoxon p = 4.31e-11, Holm-adjusted p = 8.61e-11 -> **significant (Holm p < 0.05)**

These tests compare the trained drivers across tracks. With 3 training runs per agent, they do not measure how much a new training run of the same recipe would vary.

