# Same Road, New Colors: interview questions, answered plainly

**1. What did you build, in one breath?**
I trained race-car AIs that drive from camera pixels alone in Gymnasium's CarRacing, and tested one idea: if an AI practises with a new random color scheme in every race (domain randomization), does it keep driving when the colors change to ones it has never seen? On 100 hidden test tracks in new colors, the randomized agent scored 581 against 92 for the agent trained on normal colors (better on 93 of 100 tracks, p < 0.001), and it paid a small price on normal colors: 739 vs 807.

**2. What is domain randomization, and why should anyone care?**
A model trained on one look of the world learns shortcuts that only work in that look. Here, Agent A likely learned something like "grey means road": give it a purple road on blue grass and it drives straight off at the first bend. Domain randomization varies the things that shouldn't matter (here, the colors) during training, so the model has to rely on what does matter (the shape of the road). It's a standard tool for moving robots from simulation to the real world, where lighting and textures never match the simulator. This project measures it cleanly: same tracks, same practice, only the colors differ.

**3. How did you keep the comparison fair?**
- **Same everything except colors:** the same learning method and settings, the same 5 million decisions of practice, the same 3 training seeds.
- **Same tracks:** a track seed always builds the same track, so A and B were compared on identical tracks, with paired tests.
- **Colors with their own seed:** Gymnasium's built-in color randomizer also changes the track, so I rebuilt it with a separate color seed and checked it pixel for pixel against the official game.
- **Truly unseen colors:** B practised on color seeds 1,000,000–1,999,999; the tests used seeds below that range.
- **A hidden test, fixed in advance:** the 100 test tracks were never touched until the end, and the test plan, including the verdict rules and which races become videos, was committed to git before the one-shot test.

**4. What exactly did the final test show?**
- **Main result (proven):** on unseen colors, B beat A by 490 points (95% CI 431 to 545), on 93 of 100 tracks; Wilcoxon signed-rank with Holm correction, p = 1.7e-16. Every one of B's three seeds beat every one of A's.
- **The price (proven, small):** on normal colors, B scored 68 points less than A (p = 0.0002).
- **Not immune:** B still lost 158 points on new colors compared with normal ones.
- **Caveat:** 3 training runs per agent. The tests compare these six drivers on 100 tracks; they can't say how much a fresh training run would vary.

**5. Agent B couldn't drive at first. What went wrong, and what did you do?**
With the standard RL Baselines3 Zoo settings, A learned to drive but B never did: it missed the first real bend in any colors. The likely reason (never tested): the training updates were very unstable (several times the usual size), and A coped with that while B's harder task didn't. Instead of patching the settings, I looked for what has actually reached about 900 on CarRacing in published work, and chose PPO with a Beta policy and one pedal (Petrazzini & Antonelo 2021), whose paper says learning "would not take place" without merging gas and brake. Before retraining, I wrote the rules down: if B still couldn't drive, switch automatically to a fallback method (Slaoui et al. 2020); if the new recipe made A clearly worse, stop. B learned (728 on validation), so the fallback wasn't needed.

**6. What's a Beta policy, and why one pedal?**
The AI's outputs are bounded: steering goes from −1 to 1. The usual Gaussian policy can suggest values outside that range, which get cut off, so the AI learns from actions it didn't actually take. A Beta distribution lives exactly on a bounded range, so that mismatch disappears. One pedal (above zero is gas, below zero is brake) stops the AI from pressing gas and brake at the same time. My understanding of why that matters (I didn't test it): during random early exploration, gas and brake mostly cancel out, so the car barely moves and there's little to learn from. You can see that in the untrained car's video.

**7. What do the heat maps show?**
I blurred one small patch of the AI's picture at a time and measured how much its decision changed (the blur test), and used gradients as a second method. I wrote down that a finding only counts if both methods agree.
- **Supported:** A pays more attention to the grass and background than B, on normal and unseen colors.
- **Not shown:** "B watches the road edges more". The two methods disagreed in direction, so I don't claim it.
- **The surprise:** both agents lean heavily on the dashboard at the bottom of the screen (speed, wheel spin, steering and turning gauges): 41–53% of the blur heat on 12.5% of the picture. Those gauges look the same in any colors, which may be part of why B copes (a likely reason I haven't tested; hiding the dashboard would test it).
- **Sanity check:** an untrained network's blur maps already resemble the trained ones, because blurring only matters where the picture has edges. So I read the differences, not the raw maps.

**8. Why does no agent reach 900, the official "solved" mark?**
Partly the drivers, partly a quirk I found in the game. The game is supposed to end a lap once 95% of the road is touched and the car crosses the start. But the start tile already counts as touched when the race begins, so that rule never fires: the race only ends early if every single tile is touched. My agents usually miss a few tiles, likely at corners, so they keep driving until the 20-second limit and lose 0.1 points per frame. That also corrected an earlier diagnosis of mine that blamed everything on slow driving: the new agents are fast enough. They just miss tiles.

**9. What went wrong along the way, and how did you handle it?**
- **A Windows update rebooted the PC at 2 AM** and killed a training run. I built exact resume: the simulator is rebuilt by replaying the current race from its recorded actions, so a resumed run ends bit-for-bit identical to one never stopped. I tested it by killing runs on purpose.
- **My own gate script crashed** right after making the correct decision, and stopped the overnight plan for a few minutes. My rehearsals had used stand-ins for that script. From then on I tried every tool for real (the final-test script ran end to end on scratch tracks in a separate copy of the project before the real test).
- **GPU rounding:** a check that the replayed picture gives the recorded decision failed twice, at gaps of 0.0001 and 0.001. Before widening the limit, I measured why (the GPU's fast math mode, under 0.005) and showed that a wrong picture is off by about 0.4.
- Every miss is written down in the lab notebook, not fixed quietly.

**10. What would you do next?**
- **More training seeds** (5–10) to measure how much a fresh training run varies.
- **Harder changes than colors:** textures, lighting, a different color formula, or real camera footage.
- **Test the dashboard idea:** hide the gauges and see whether B loses its robustness.
- **Finish laps:** a training signal for touching the last few tiles, while still scoring with the official reward.
- **Compare with the fallback method** (Slaoui et al.), which trains on normal colors but teaches the network to see recolored copies the same way.
