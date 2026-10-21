# Speed diagnosis: A_seed142_beta, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 63.5
- races_fast_enough_on_average: 83
- races_out_of_time: 90
- races_out_of_time_with_90pct_covered: 71
- overall_mean_gas: 0.3
- overall_pct_full_gas: 1.02
- highest_gas_used: 0.992
- top_speed_seen: 100.0

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           51.05 |        63.27 |       88.76 |       0.33 |           0.85 |          4.2  |                      7.88 |
| gentle (15-45)     |           17.01 |        70.38 |       92.69 |       0.31 |           0.46 |          6.32 |                      6.59 |
| medium (45-90)     |           20.53 |        65.13 |       92.7  |       0.3  |           1.67 |         11.61 |                      4.84 |
| sharp (>90)        |           11.41 |        50.38 |       88.93 |       0.15 |           1.42 |         35.52 |                      6.38 |
