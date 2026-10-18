# Speed diagnosis: A_seed42_entropy, validation_normal

- races: 100
- replays_matching: 100/100
- mean_needed_speed: 51.7
- mean_achieved_speed: 51.9
- races_fast_enough_on_average: 53
- races_out_of_time: 78
- races_out_of_time_with_90pct_covered: 55
- overall_mean_gas: 0.086
- overall_pct_full_gas: 0.0
- highest_gas_used: 0.506
- top_speed_seen: 78.3

By how sharp the road ahead is (all frames of all races):

| turn               |   share_of_time |   mean_speed |   p90_speed |   mean_gas |   pct_full_gas |   pct_braking |   pct_all_wheels_off_road |
|:-------------------|----------------:|-------------:|------------:|-----------:|---------------:|--------------:|--------------------------:|
| straight (<15 deg) |           52.6  |        49.45 |       66.61 |       0.09 |              0 |          0.44 |                      3.91 |
| gentle (15-45)     |           17.55 |        54.93 |       66.1  |       0.08 |              0 |          1.11 |                      7.55 |
| medium (45-90)     |           20.74 |        53.65 |       66.36 |       0.09 |              0 |          0.78 |                      7.85 |
| sharp (>90)        |            9.1  |        56.66 |       66.66 |       0.08 |              0 |          3.41 |                     17.42 |
