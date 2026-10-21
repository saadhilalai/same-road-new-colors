# Speed diagnosis: A_seed42_beta, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 63.9
- races_fast_enough_on_average: 89
- races_out_of_time: 89
- races_out_of_time_with_90pct_covered: 64
- overall_mean_gas: 0.294
- overall_pct_full_gas: 0.51
- highest_gas_used: 0.972
- top_speed_seen: 100.0

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           52.13 |        62.28 |       87    |       0.31 |           0.52 |          3.83 |                      9.87 |
| gentle (15-45)     |           17.88 |        66.47 |       90.58 |       0.28 |           0.3  |          9.8  |                     13.27 |
| medium (45-90)     |           20.32 |        66.6  |       91.43 |       0.28 |           0.47 |          7.22 |                      9.26 |
| sharp (>90)        |            9.66 |        62.71 |       89.17 |       0.26 |           0.98 |         13.51 |                     21.6  |
