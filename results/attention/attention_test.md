# Step 4: what the AI looks at (analysis tracks, final copies, recipe beta)

Rule: PROGRESS.md, Step 4 (written before any A-vs-B result). Focus = share of the heat map in a region divided by the share of the picture the region covers (1 = its size's worth). Per track: mean of 3 seeds.

## Focus and share per region: blur test (main)

| agent   | colors   |   races with moments |   road edge focus |   road focus |   off-road focus |   car focus |   dashboard focus | road edge share   | road share   | off-road share   | car share   | dashboard share   |
|:--------|:---------|---------------------:|------------------:|-------------:|-----------------:|------------:|------------------:|:------------------|:-------------|:-----------------|:------------|:------------------|
| A       | normal   |                  300 |              1.13 |         0.84 |             0.54 |        1.85 |              3.31 | 12.1%             | 12.2%        | 33.3%            | 1.1%        | 41.3%             |
| A       | unseen   |                  300 |              1.16 |         1.08 |             0.48 |        2.41 |              3.89 | 6.1%              | 8.5%         | 35.3%            | 1.4%        | 48.7%             |
| B       | normal   |                  300 |              0.82 |         0.75 |             0.43 |        1.75 |              4.26 | 8.6%              | 10.7%        | 26.4%            | 1.0%        | 53.2%             |
| B       | unseen   |                  300 |              0.95 |         0.93 |             0.43 |        2.22 |              4.08 | 8.5%              | 11.2%        | 28.0%            | 1.3%        | 51.0%             |

## Focus and share per region: gradient check

| agent   | colors   |   races with moments |   road edge focus |   road focus |   off-road focus |   car focus |   dashboard focus | road edge share   | road share   | off-road share   | car share   | dashboard share   |
|:--------|:---------|---------------------:|------------------:|-------------:|-----------------:|------------:|------------------:|:------------------|:-------------|:-----------------|:------------|:------------------|
| A       | normal   |                  300 |              1.12 |         0.95 |             1.02 |        1.01 |              0.88 | 11.9%             | 13.8%        | 62.7%            | 0.6%        | 11.0%             |
| A       | unseen   |                  300 |              1.12 |         1.15 |             0.99 |        1.48 |              0.82 | 6.5%              | 9.9%         | 72.5%            | 0.9%        | 10.3%             |
| B       | normal   |                  300 |              1.29 |         1.27 |             0.78 |        1.2  |              1.53 | 13.6%             | 18.3%        | 48.2%            | 0.7%        | 19.2%             |
| B       | unseen   |                  300 |              1.28 |         1.29 |             0.77 |        1.35 |              1.64 | 11.9%             | 16.5%        | 50.3%            | 0.8%        | 20.5%             |

## The four tests (Wilcoxon, two-sided, Holm over 4, per method)

- 1. Normal colors: road-edge focus, B minus A [blur]: 0.817 vs 1.132, difference -0.316 (95% CI -0.329 to -0.302); first higher on 0 of 100 tracks; p = 3.9e-18, Holm p = 1.56e-17 -> significant, opposite
- 2. Unseen colors: road-edge focus, B minus A [blur]: 0.953 vs 1.157, difference -0.205 (95% CI -0.278 to -0.131); first higher on 25 of 100 tracks; p = 2.41e-07, Holm p = 2.41e-07 -> significant, opposite
- 3. Normal colors: off-road focus, A minus B [blur]: 0.539 vs 0.426, difference +0.113 (95% CI +0.107 to +0.120); first higher on 99 of 100 tracks; p = 4.02e-18, Holm p = 1.56e-17 -> significant, as guessed
- 4. Unseen colors: off-road focus, A minus B [blur]: 0.482 vs 0.429, difference +0.053 (95% CI +0.037 to +0.069); first higher on 78 of 100 tracks; p = 8.11e-09, Holm p = 1.62e-08 -> significant, as guessed
- 1. Normal colors: road-edge focus, B minus A [gradient]: 1.286 vs 1.117, difference +0.168 (95% CI +0.162 to +0.175); first higher on 100 of 100 tracks; p = 3.9e-18, Holm p = 1.56e-17 -> significant, as guessed
- 2. Unseen colors: road-edge focus, B minus A [gradient]: 1.283 vs 1.123, difference +0.161 (95% CI +0.129 to +0.194); first higher on 80 of 100 tracks; p = 6.03e-13, Holm p = 6.03e-13 -> significant, as guessed
- 3. Normal colors: off-road focus, A minus B [gradient]: 1.016 vs 0.777, difference +0.239 (95% CI +0.234 to +0.243); first higher on 100 of 100 tracks; p = 3.9e-18, Holm p = 1.56e-17 -> significant, as guessed
- 4. Unseen colors: off-road focus, A minus B [gradient]: 0.992 vs 0.771, difference +0.220 (95% CI +0.205 to +0.236); first higher on 99 of 100 tracks; p = 4.14e-18, Holm p = 1.56e-17 -> significant, as guessed

## Verdicts (supported = both methods significant in the guessed direction)

- 1. Normal colors: road-edge focus, B minus A: **not shown**
- 2. Unseen colors: road-edge focus, B minus A: **not shown**
- 3. Normal colors: off-road focus, A minus B: **supported**
- 4. Unseen colors: off-road focus, A minus B: **supported**

## Checks

- How well the two methods agree (rank correlation of the two maps per picture, mean): A normal 0.33, A unseen 0.24, B normal 0.34, B unseen 0.37
- Sanity check, A seed 42, normal colors, same pictures: trained vs untrained network, rank correlation of the maps: blur 0.71, gradient 0.57 (near 1 would mean the maps only trace the picture).
- Untrained network's focus (blur): road edge 2.09, road 1.87, off-road 0.57, car 2.96, dashboard 1.10; trained A seed 42: road edge 1.10, road 0.83, off-road 0.54, car 2.00, dashboard 3.32
- Largest gap between a rebuilt picture's action and the recorded one: 1.5e-03 (limit 1e-2).

