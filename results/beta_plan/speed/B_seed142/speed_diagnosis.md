# Speed diagnosis: B_seed142_beta, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 56.4
- races_fast_enough_on_average: 72
- races_out_of_time: 100
- races_out_of_time_with_90pct_covered: 52
- overall_mean_gas: 0.302
- overall_pct_full_gas: 2.86
- highest_gas_used: 0.983
- top_speed_seen: 100.0

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           52.42 |        54    |       78.78 |       0.32 |           2.87 |          6.06 |                     14.67 |
| gentle (15-45)     |           18.28 |        60.38 |       84.12 |       0.29 |           2.86 |          8.89 |                     14.8  |
| medium (45-90)     |           20.14 |        58.68 |       83.5  |       0.28 |           2.34 |          6.89 |                     12.74 |
| sharp (>90)        |            9.16 |        56.9  |       83.33 |       0.26 |           3.94 |          9.04 |                     19.99 |
