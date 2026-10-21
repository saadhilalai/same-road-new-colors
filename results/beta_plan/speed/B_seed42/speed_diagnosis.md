# Speed diagnosis: B_seed42_beta, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 53.1
- races_fast_enough_on_average: 61
- races_out_of_time: 84
- races_out_of_time_with_90pct_covered: 59
- overall_mean_gas: 0.258
- overall_pct_full_gas: 0.79
- highest_gas_used: 0.992
- top_speed_seen: 90.5

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           53.54 |        50.43 |       74.5  |       0.28 |           0.69 |          8.25 |                     15.37 |
| gentle (15-45)     |           18.22 |        56.25 |       78.24 |       0.26 |           1.41 |          9.17 |                     13.12 |
| medium (45-90)     |           20.31 |        55.39 |       78.47 |       0.24 |           0.58 |          7.35 |                     11.96 |
| sharp (>90)        |            7.92 |        59.28 |       80.08 |       0.18 |           0.65 |         10.52 |                     14.91 |
