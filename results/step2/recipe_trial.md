# Recipe trial: agent A, seed 42, final copies, 100 validation tracks (normal colors)

| recipe   | change              |   races |   mean_score |   std_score | solved_900   |   mean_track_covered_pct |   lap_completed_pct |   lap_finished_gym_pct |   off_track_pct |
|:---------|:--------------------|--------:|-------------:|------------:|:-------------|-------------------------:|--------------------:|-----------------------:|----------------:|
| entropy  | {"ent_coef": 0.01}  |     100 |        781.8 |       185.5 | False        |                     88.5 |                  16 |                     16 |               6 |
| kl_limit | {"target_kl": 0.05} |     100 |        692.7 |       215.1 | False        |                     79.1 |                   8 |                      8 |               0 |

Paired comparison on the same 100 tracks (entropy minus kl_limit):
- mean difference +89.2 points; entropy better on 66 tracks, kl_limit better on 34, tied on 0
- Wilcoxon signed-rank test: statistic 1518.0, p = 0.0005354 (significant at the 5% level)

**Winner by the pre-written rule: `entropy`** (higher mean; minimum needed 500).
