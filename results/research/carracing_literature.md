# What has reached ~900 on CarRacing: published evidence (collected 2026-10-19)

Collected to pick a training recipe from evidence instead of patching the current one (director's request, 2026-10-19).
Sources were fetched and read; the key numbers marked "checked" were re-read by me in the original source.
"Unverified" = could not be confirmed on the source page. Published scores are mostly on CarRacing-v0; the reward,
time limit and 900 "solved" mark are the same in v3, but lap-ending rules and evaluation setups differ, so they are
reference points, not exact comparisons.

## PPO and other gradient-based RL from pixels

| Source | Method | Score (how measured) | Training | Main settings |
|---|---|---|---|---|
| RL Zoo benchmark (checked) | PPO, Zoo recipe = **our starting recipe** (grayscale there) | **153.9 +/- 74.5**, deterministic, 179 episodes, v0 | 4M steps, 1 run | gSDE Gaussian, lr 1e-4, clip 0.2, frame skip 2, 64x64, 2 frames |
| RL Zoo benchmark (checked) | RecurrentPPO (LSTM memory) | **862.5 +/- 97.3**, deterministic, 156 episodes, v0 | 4M steps, 1 run | same Zoo pipeline, grayscale 64x64, 2 frames |
| Hugging Face model cards | Zoo PPO / ppo_lstm, v0 | 175.0 / 880.4 (self-reported) | 4M | as above |
| Hugging Face community cards | Zoo PPO / ppo_lstm, **v3** | 248.2 / 384.9 (self-reported) | 4M | as above |
| Petrazzini & Antonelo 2021, arXiv 2111.02202 (checked) | **PPO with a Beta policy, gas+brake merged into one pedal** | 5 seeds, 100 consecutive episodes each: **mean 816.1 with the best action (deterministic); all 5 seeds above 900 with sampled actions**. Best seed 913 +/- 26 (gym leaderboard). Gaussian policy, same settings: 370.4 deterministic / 890.5 sampled | 5M steps, 8 envs | grayscale 84x84, 4 frames; lr 2.5e-4 annealed linearly to 0; clip 0.1; 10 epochs; minibatch 64; 500 steps per env per update; gamma 0.99; GAE 0.95; entropy 0.01; value 0.5; 3 conv layers + 512 units. **Original reward, no early stopping.** Says that without merging gas and brake "learning to control the vehicle would not take place" |
| JinayJain/deep-racing (gym leaderboard) | PPO + Beta, one pedal | 909 +/- 10, 100 episodes, sampled actions | ~5.1M steps, 1 env | grayscale 96x96, 4 frames, lr 1e-3, clip 0.1; rewards clipped in training; **episode cut after 30 steps without progress, also during evaluation** (a trick) |
| xtma/pytorch_car_caring | PPO + Beta | README: ~890 running average of a **shaped** training reward; a user's re-test without shaping: **~860 over 100 episodes** | ~1900 episodes | action repeat 8, reward shaping, early termination |
| Rafael1s (gym leaderboard) | same code as xtma | claimed 901 (training average of shaped reward); leaderboard maintainers re-tested: **820** | 2760 episodes, ~7 h | reward shaping |
| AMD-RIPS 2018 (gym leaderboard) | DQN, 5 buttons | 906.7 +/- 23.6 on 100 tracks | trained on 3 fixed tracks | grayscale 84x84, 4 frames |
| jperod (gym leaderboard) | DQN, 5 buttons | 905 +/- 24 | ~1.6M steps, ~30 h on a GTX 1050 Ti (unverified) | **early stop after 8 bad steps, also at test** (a trick) |
| Ceudan/Car-Racing (leaderboard, v1) | two PPO + Beta agents | 917, 100 episodes | 925 episodes | custom cropping, speed from pixels, early termination on grass |
| Forum post (Gemini dev forum), **v3 with random colors** | DQN with many add-ons | 831.5 +/- 108.5, 100 episodes (self-reported, unverified) | "a few days or 1 week" on a MacBook Air M2 | custom contrast preprocessing |

No library benchmark (CleanRL, Tianshou, Sample Factory, Dopamine, EnvPool, rl_games) publishes a CarRacing score.
No verified SAC / DrQ / CURL score on CarRacing from pixels was found.

## Evolution, world models, attention (not PPO)

Time on this PC = frames / ~5.5M frames per hour (measured: one simulator 210 frames/s; 10 independent simulators
about 1,500 frames/s). Simulator time only, so these are best cases.

| Source | Score | Training | Time on this PC |
|---|---|---|---|
| World Models, Ha & Schmidhuber 2018 | 906 +/- 21, 100 random tracks | CMA-ES 64 x 16 rollouts x 1800 generations = ~1.84B frames; 64-core cloud VM | ~14 days per run |
| Genetic algorithm + world model, Risi & Stanley 2019 | 903 +/- 72 | ~324k episodes (at most ~324M frames, early stop when no progress); 32 cores, ~2 days | up to ~2.5 days per run |
| Attention agent, Tang, Nguyen & Ha 2020 | 914 +/- 15, 100 episodes | CMA-ES 256 x 16 rollouts x ~1000 iterations = up to ~4.1B frames | up to ~1 month per run |
| AttentionNeuron, Tang & Ha 2021 | 801 +/- 147 | 256 CPUs for 10 days (~16B frames) | months |
| Dreamer-style (community repos, v3) | ~900 claimed, not a 100-random-track evaluation | ~0.6-1M steps, about a day of GPU (unverified) | maybe 1 day, but evidence weak |

## Robustness to colors (closest to the project question)

| Source | Setup | Result |
|---|---|---|
| **Slaoui, Clements, Foerster & Toth 2020** (ICLR 2020 workshop, arXiv 1910.10537; checked) | PPO, CarRacing-v0, background color randomized over the whole RGB range, 5 seeds | Normal-color agent: 554 normal / 60 all colors. **Plain random-color agent: -17 / -35: "the randomized agent ... fail[s] to learn on this large randomization space."** Their fix (train on normal colors, show a recolored copy of the same image, penalize the AI when its internal picture of the two differs), lambda 50: **640 / 553** |
| Tang, Nguyen & Ha 2020, Table 4 | trained on normal colors, tested with slightly shifted lane/grass colors | PPO 865 -> 505 +/- 464; World Models 901 -> 655; GA 859 -> 442; attention agent 914 -> 866 |
| Xing et al. 2021 (LUSR, AAAI) | PPO 10M steps, unseen background/car/road colors | special pretrained image encoder: 805 -> 679 / 801; CURL: 749 -> -113 / -69 |
| Jian et al. 2024 (R3L) | PPO 5M steps, grass colors | one-color agents 829-872; agent trained on green/red/yellow scored 158 on unseen blue |
| Tang & Ha 2021 | new background images | NetRand (random-convolution augmentation) only 480 on the normal game: "too difficult for NetRand" |

## Version differences (v0 -> v3)
- v1 (gym 0.24): new lap-ending rule (>95% of tiles + back at the start), domain randomization added, pygame rendering.
- v2 (gym 0.25): finishing a lap ends the race as termination, not truncation (mixed up in v1).
- v3 (Gymnasium 1.0): the same fix again (completing the track terminates, not truncates).
- Unchanged: -0.1 per frame, +1000/N per new tile, -100 for leaving the map, 1000-frame limit, 96x96 image, 900 = solved.
- Open Gymnasium issue #1269: `lap_complete_percent` appears to be ignored in v3 (does not affect our strict 100%-of-tiles lap rule).

## Sources
- https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/benchmark.md
- https://arxiv.org/abs/2111.02202 (Petrazzini & Antonelo, PPO with Beta distribution)
- https://github.com/openai/gym/wiki/Leaderboard
- https://github.com/JinayJain/deep-racing , https://github.com/xtma/pytorch_car_caring , https://github.com/jperod , https://github.com/Ceudan/Car-Racing
- https://discuss.ai.google.dev/t/open-source-avg-800-dqn-agent-on-car-racing-v3-random-environment/89982
- https://worldmodels.github.io/ , https://blog.otoro.net/2018/06/09/world-models-experiments/
- https://arxiv.org/abs/1906.08857 (Risi & Stanley) , https://arxiv.org/abs/2003.08165 (attention agent) , https://arxiv.org/abs/2109.02869 (AttentionNeuron)
- https://arxiv.org/abs/1910.10537 (Slaoui et al.) , https://github.com/uncharted-technologies/robust-domain-randomization
- https://arxiv.org/abs/2102.05714 (LUSR) , https://arxiv.org/abs/2404.12917 (R3L)
- https://github.com/openai/gym/releases/tag/0.24.0 , https://gymnasium.farama.org/main/release_notes/gym_v0.25.0/ ,
  https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.0.0 , https://github.com/Farama-Foundation/Gymnasium/issues/1269
