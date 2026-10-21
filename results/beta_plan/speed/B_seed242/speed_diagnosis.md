# Speed diagnosis: B_seed242_beta, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 59.9
- races_fast_enough_on_average: 84
- races_out_of_time: 71
- races_out_of_time_with_90pct_covered: 53
- overall_mean_gas: 0.302
- overall_pct_full_gas: 3.24
- highest_gas_used: 0.995
- top_speed_seen: 92.4

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           51.8  |        57.09 |       77.25 |       0.34 |           4.12 |          3.57 |                      8.62 |
| gentle (15-45)     |           18.46 |        62.74 |       79.31 |       0.29 |           3.1  |          6.66 |                     11.76 |
| medium (45-90)     |           20.31 |        62.59 |       79.49 |       0.27 |           1.9  |          8.38 |                      6.69 |
| sharp (>90)        |            9.43 |        62.28 |       80.63 |       0.19 |           1.51 |         16.19 |                     13.52 |
