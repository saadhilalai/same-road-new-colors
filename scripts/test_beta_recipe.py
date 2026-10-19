"""
Checks for recipe "beta" and Agent B's fallback (Slaoui's recolored copies), run before any training with them.

    python -m scripts.test_beta_recipe

1. One pedal: [steer, pedal] -> [steer, gas, brake] is right, also at the edges.
2. Beta distribution: log-probability, entropy and best action (mean) match torch's Beta on [0, 1], stretched to [-1, 1].
3. Recolored copy: drawing it never changes the game, and it is pixel-identical (also after shrinking to 64x64)
   to the same race actually played in those colors.
4. Slaoui agent: it acts on the normal images only (changing the copies changes nothing), in a plain agent's layout.
5. With penalty 0, SlaouiPPO trains bit-for-bit like SB3's PPO; with the penalty on, the weights differ.
6. Evaluation of a one-pedal agent records the game's 3 controls, and replaying them reproduces every race exactly.
Exit code 0 only if everything passes.
"""

import sys

import numpy as np
import torch as th

import config
from racecar.beta_policy import BetaCnnPolicy, BetaDistribution, NormalViewCNN, SlaouiPPO
from racecar.envs import make_env, one_pedal_to_game
from racecar.evaluate import evaluate
from racecar.replay import check_matches, replay
from racecar.train_utils import build_beta_ppo, build_vec_env, view_of

TRACK = config.SCRATCH_TRACK_START + 7
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)


def driving_actions(n, seed=0):
    """Some gas, some steering, some braking."""
    rng = np.random.default_rng(seed)
    acts = []
    for i in range(n):
        acts.append(np.array([np.sin(i / 15), 0.6 if i % 40 < 30 else 0.0, 0.8 if i % 40 >= 36 else 0.0], np.float32))
        acts[-1][0] += rng.normal(0, 0.1)
    return acts


def test_one_pedal():
    a = np.array([[0.3, 0.7], [-1.0, -0.4], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]], np.float32)
    want = np.array([[0.3, 0.7, 0], [-1, 0, 0.4], [1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32)
    got = one_pedal_to_game(a)
    check("one pedal -> steer/gas/brake", got.dtype == np.float32 and np.allclose(got, want), str(got.tolist()))
    env = make_env(one_pedal=True)
    check("one-pedal env takes 2 numbers", env.action_space.shape == (2,), str(env.action_space))
    env.close()


def test_beta():
    th.manual_seed(0)
    d = BetaDistribution(2)
    params = th.randn(5000, 4) * 2
    d.proba_distribution(params)
    alpha, beta = (1 + th.nn.functional.softplus(params)).chunk(2, dim=-1)
    ref = th.distributions.Beta(alpha, beta)
    a = d.sample()
    check("samples inside [-1, 1]", bool((a.abs() < 1).all()), f"min {a.min():.6f}, max {a.max():.6f}")
    lp_ref = (ref.log_prob((a + 1) / 2) - np.log(2)).sum(-1)
    check("log-probability = torch Beta, stretched", th.allclose(d.log_prob(a), lp_ref, atol=1e-4),
          f"max diff {(d.log_prob(a) - lp_ref).abs().max():.2e}")
    check("entropy = torch Beta + log 2 per number", th.allclose(d.entropy(), (ref.entropy() + np.log(2)).sum(-1)))
    check("best action = Beta mean, stretched", th.allclose(d.mode(), 2 * alpha / (alpha + beta) - 1))
    # Monte Carlo: average of many samples from one distribution = its mean
    d.proba_distribution(th.tensor([[0.5, -1.0, 2.0, 0.3]]).repeat(200_000, 1))
    mc = d.sample().mean(0)
    check("sample average matches the mean", th.allclose(mc, d.mode()[0], atol=5e-3), f"{mc.tolist()} vs {d.mode()[0].tolist()}")
    # Edges: an action of exactly -1 or 1 still has a finite log-probability
    d.proba_distribution(th.zeros(1, 4))
    check("finite log-probability at the edges", bool(th.isfinite(d.log_prob(th.tensor([[-1.0, 1.0]]))).all()))


def test_recolored_copy():
    color_seed = 1_234_567
    acts = driving_actions(300)
    plain, drawn, colored = make_env(ai_view=False), make_env(ai_view=False), make_env(ai_view=False)
    o1, _ = plain.reset(seed=TRACK)
    o2, _ = drawn.reset(seed=TRACK)
    o3, _ = colored.reset(seed=TRACK, options={"color_seed": color_seed})
    same_game, worst_copy_diff, n_copy_diff = True, 0, 0
    first_copy = drawn.unwrapped.render_in_colors(color_seed)
    worst_copy_diff = int(np.abs(first_copy.astype(int) - o3.astype(int)).max())
    for a in acts:
        r1 = plain.step(a)
        copy_ = drawn.unwrapped.render_in_colors(color_seed)   # drawn every frame here: a hard test
        r2 = drawn.step(a)
        r3 = colored.step(a)
        same_game &= np.array_equal(r1[0], r2[0]) and r1[1] == r2[1] and r1[2:4] == r2[2:4]
        d = np.abs(drawn.unwrapped.render_in_colors(color_seed).astype(int) - r3[0].astype(int)).max()
        worst_copy_diff = max(worst_copy_diff, int(d))
        n_copy_diff += int(d > 0)
        if r1[2] or r1[3]:
            break
    tiles = plain.unwrapped.tile_visited_count
    check("drawing copies never changes the game", same_game, f"{len(acts)} actions, {tiles} tiles visited")
    check("copy = the same race played in those colors", worst_copy_diff == 0,
          f"largest pixel difference {worst_copy_diff}, frames with any difference {n_copy_diff}")
    check("copy differs from the normal image", not np.array_equal(copy_, r1[0]))
    for e in (plain, drawn, colored):
        e.close()

    # The wrapper's 64x64 copy = the AI's 64x64 view of the race played in those colors
    with_copy, colored = make_env(copies="recolor"), make_env()
    o, _ = with_copy.reset(seed=TRACK)
    s = with_copy.copy_color_seed
    c, _ = colored.reset(seed=TRACK, options={"color_seed": s})
    plain_view = make_env()
    p, _ = plain_view.reset(seed=TRACK)
    same = np.array_equal(o[..., 3:], c) and np.array_equal(o[..., :3], p)
    for a in acts[:150]:
        o, *_ = with_copy.step(a)
        c, *_ = colored.step(a)
        p, *_ = plain_view.step(a)
        same &= np.array_equal(o[..., 3:], c) and np.array_equal(o[..., :3], p)
    check("64x64 copy = AI view of the recolored race; normal half = plain view", same, f"copy color seed {s}")
    for e in (with_copy, colored, plain_view):
        e.close()


def small_venv(copies, seed=3):
    return build_vec_env(2, seed, "normal", True, False, one_pedal=True, copies=copies)


def test_slaoui_agent():
    venv, plain = small_venv(copies=True), small_venv(copies=False)
    obs, pobs = venv.reset(), plain.reset()
    for _ in range(30):
        obs, *_ = venv.step(np.array([[0.0, 0.5]] * 2, np.float32))
        pobs, *_ = plain.step(np.array([[0.0, 0.5]] * 2, np.float32))
    normal = obs[..., [0, 1, 2, 6, 7, 8]]
    check("normal channels = a plain agent's images", np.array_equal(normal, pobs))
    check("copies really are recolored", not np.array_equal(obs[..., [3, 4, 5, 9, 10, 11]], normal))
    model = build_beta_ppo(venv, 0, None, "beta_slaoui")   # (this re-seeds the simulators, so it comes after)
    fe = model.policy.features_extractor
    check("Slaoui network reads normal channels 0-2, 6-8; copies 3-5, 9-11",
          fe.normal_idx.tolist() == [0, 1, 2, 6, 7, 8] and fe.copy_idx.tolist() == [3, 4, 5, 9, 10, 11],
          f"{fe.normal_idx.tolist()} / {fe.copy_idx.tolist()}")
    check("view read from the model", view_of(model) == {"one_pedal": True, "copies": True}, str(view_of(model)))
    noisy = obs.copy()
    noisy[..., [3, 4, 5, 9, 10, 11]] = np.random.default_rng(0).integers(0, 256, noisy[..., :6].shape)
    a1, _ = model.predict(obs, deterministic=True)
    a2, _ = model.predict(noisy, deterministic=True)
    t = lambda o: th.as_tensor(o.transpose(0, 3, 1, 2), device=model.device)
    v1, v2 = model.policy.predict_values(t(obs)), model.policy.predict_values(t(noisy))
    check("acting ignores the copies", np.array_equal(a1, a2) and th.equal(v1, v2))
    gap = model.policy.copy_feature_gap(t(obs).float()).item()
    check("penalty is positive for recolored copies", gap > 0, f"{gap:.3e}")
    venv.close()
    plain.close()


def test_slaoui_ppo_equals_ppo():
    def trained(cls, **extra):
        venv = small_venv(copies=True, seed=5)
        kw = dict(n_steps=64, batch_size=32, n_epochs=2, ent_coef=0.01, clip_range=0.1, seed=11, device=config.DEVICE,
                  policy_kwargs=dict(features_extractor_class=NormalViewCNN, net_arch=dict(pi=[64], vf=[64])))
        th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = True, False
        m = cls(BetaCnnPolicy, venv, **kw, **extra)
        m.learn(256)
        venv.close()
        return [p.detach().cpu().clone() for p in m.policy.parameters()]

    from stable_baselines3 import PPO
    base = trained(PPO)
    zero = trained(SlaouiPPO, copy_penalty=0.0)
    on = trained(SlaouiPPO, copy_penalty=50.0)
    check("SlaouiPPO with penalty 0 = SB3 PPO, bit for bit", all(th.equal(a, b) for a, b in zip(base, zero)))
    check("penalty on changes the training", not all(th.equal(a, b) for a, b in zip(base, on)))


def test_eval_replay():
    for recipe, copies in (("beta", False), ("beta_slaoui", True)):
        venv = small_venv(copies=copies)
        model = build_beta_ppo(venv, 0, None, recipe)
        model.learn(1000)   # barely trained: it wobbles, so the replay has real work to do
        venv.close()
        races = [(TRACK, None), (TRACK + 1, 5003)]
        df, actions = evaluate(model, races, record_actions=True)
        shapes = {a.shape[1] for a in actions.values()}
        check(f"{recipe}: recorded actions are the game's 3 controls", shapes == {3}, str(shapes))
        ok = True
        for _, row in df.iterrows():
            acts = actions[(row["track_seed"], row["color_seed"])]
            try:
                check_matches(replay(row["track_seed"], row["color_seed"], acts, render=False), row)
            except AssertionError as e:
                ok = False
                print(e)
        check(f"{recipe}: replays reproduce every race", ok, ", ".join(f"{s:.1f}" for s in df["score"]))


if __name__ == "__main__":
    test_one_pedal()
    test_beta()
    test_recolored_copy()
    test_slaoui_agent()
    test_slaoui_ppo_equals_ppo()
    test_eval_replay()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)
