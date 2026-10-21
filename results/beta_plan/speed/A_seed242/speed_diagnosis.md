# Speed diagnosis: A_seed242_beta, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 67.1
- races_fast_enough_on_average: 94
- races_out_of_time: 83
- races_out_of_time_with_90pct_covered: 66
- overall_mean_gas: 0.317
- overall_pct_full_gas: 0.64
- highest_gas_used: 0.979
- top_speed_seen: 100.0

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           51.87 |        65.24 |       90.5  |       0.34 |           0.63 |          3.8  |                     10.32 |
| gentle (15-45)     |           17.38 |        72.79 |       94.99 |       0.32 |           0.48 |          4.83 |                     10.36 |
| medium (45-90)     |           20.98 |        68.12 |       93.97 |       0.31 |           0.89 |          8.58 |                     10.98 |
| sharp (>90)        |            9.77 |        64.06 |       93.21 |       0.18 |           0.39 |         28.89 |                      7.61 |
