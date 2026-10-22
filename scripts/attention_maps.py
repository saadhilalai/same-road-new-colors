"""
Step 4: heat maps of what the AI looks at, for every agent, seed and color set on the analysis tracks.
Replays the Step 3 races exactly (racecar.attention_frames, several in parallel), makes the blur-test and gradient
heat maps of the sampled moments (racecar.attention, on the GPU) and measures how much heat lands in each region.

    python -m scripts.attention_maps                                   # all agents, seeds, normal + unseen colors
    python -m scripts.attention_maps --agents A --seeds 42 --colors normal

Writes results/attention/<run name>_<tracks>_<colors>.csv (one row per race; finished ones are skipped, so a stopped
run just continues), results/attention/sanity_untrained.csv (the untrained network on A seed 42's normal-color images)
and runs/attention_examples/*.npz (a few maps per race for figures; not in git).
Stops with an error if a replay doesn't match its recorded race or a rebuilt image doesn't give the recorded action.
"""

import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO

import config
from racecar.attention import blur_maps, gradient_maps, patch_masks, region_shares
from racecar.attention_frames import REGIONS, replay_for_attention
from racecar.envs import one_pedal_to_game

OUT = config.RESULTS_DIR / "attention"
EXAMPLES = config.RUNS_DIR / "attention_examples"
METHODS = ("blur", "gradient")
ACTION_TOLERANCE = 1e-3      # rebuilt images must give the recorded action: GPU rounding reaches 2e-4, wrong images ~0.4
EXAMPLE_TRACKS = 5           # keep the maps of the first 5 tracks for figures


def load_model(agent, seed, recipe, checkpoint="final"):
    model = PPO.load(config.RUNS_DIR / config.run_name(agent, seed, recipe) / "checkpoints" / f"model_{checkpoint}.zip",
                     device=config.DEVICE)
    model.policy.set_training_mode(False)
    return model


def maps_for(policy, stacks, masks):
    return {"blur": blur_maps(policy, stacks, masks), "gradient": gradient_maps(policy, stacks, config.DEVICE)}


def rank_correlation(a, b):
    """Spearman correlation between two heat maps, per image."""
    ra = a.reshape(len(a), -1).argsort(1).argsort(1).astype(np.float64)
    rb = b.reshape(len(b), -1).argsort(1).argsort(1).astype(np.float64)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    return (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))


def race_row(maps, fractions):
    """Shares, areas and focus (share / area) per method and region, averaged over the race's moments."""
    area = fractions.astype(np.float32).mean(axis=(2, 3)).mean(axis=0)
    row = {f"area_{r}": float(a) for r, a in zip(REGIONS, area)}
    for method, m in maps.items():
        share = np.nanmean(region_shares(m, fractions), axis=0)
        for r, sh, a in zip(REGIONS, share, area):
            row[f"share_{method}_{r}"] = float(sh)
            row[f"focus_{method}_{r}"] = float(sh / a) if a > 0 else np.nan
    return row


def run_one(agent, seed, recipe, colors, track_set, pool, masks, sanity_model=None):
    run = config.run_name(agent, seed, recipe)
    recorded = pd.read_csv(config.RESULTS_DIR / "evals" / run / f"{track_set}_{colors}.csv")
    actions = np.load(config.RUNS_DIR / run / "actions" / f"{track_set}_{colors}.npz")
    model = load_model(agent, seed, recipe)
    tasks = [dict(track_seed=r.track_seed, color_seed=r.color_seed, actions=actions[f"{r.track_seed}_{r.color_seed}"])
             for r in recorded.itertuples()]
    rows, sanity_rows, examples = [], [], {}
    for rec, res in zip(recorded.itertuples(), pool.map(replay_for_attention, tasks)):
        if not (abs(res["score"] - rec.score) < 1e-3 and res["frames"] == rec.frames
                and res["tiles_visited"] == rec.tiles_visited):
            raise AssertionError(f"{run} track {rec.track_seed}: replay {res['score']:.3f} / {res['frames']} frames "
                                 f"vs recorded {rec.score:.3f} / {rec.frames}")
        row = {"agent": agent, "seed": seed, "recipe": recipe, "track_seed": rec.track_seed,
               "color_seed": rec.color_seed, "score": rec.score, "moments": len(res["decisions"])}
        if len(res["decisions"]) == 0:
            rows.append(row)
            continue
        predicted, _ = model.predict(res["stacks"], deterministic=True)
        gap = float(np.abs(one_pedal_to_game(predicted) - res["actions"]).max())
        if gap > ACTION_TOLERANCE:
            raise AssertionError(f"{run} track {rec.track_seed}: rebuilt images give a different action (gap {gap:.2e})")
        maps = maps_for(model.policy, res["stacks"], masks)
        row.update(max_action_gap=gap, **race_row(maps, res["fractions"]),
                   methods_agreement=float(np.nanmean(rank_correlation(maps["blur"], maps["gradient"]))))
        rows.append(row)
        if sanity_model is not None:
            untrained = maps_for(sanity_model.policy, res["stacks"], masks)
            s = {"track_seed": rec.track_seed, **{k: v for k, v in race_row(untrained, res["fractions"]).items()
                                                  if not k.startswith("area_")}}
            for method in METHODS:
                s[f"trained_vs_untrained_{method}"] = float(np.nanmean(rank_correlation(maps[method], untrained[method])))
            sanity_rows.append(s)
        if len(examples) < EXAMPLE_TRACKS:
            keep = slice(None, None, max(1, len(res["decisions"]) // 6))
            examples[str(rec.track_seed)] = dict(decisions=res["decisions"][keep], stacks=res["stacks"][keep],
                                                 fractions=res["fractions"][keep],
                                                 **{m: maps[m][keep].astype(np.float32) for m in METHODS})
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EXAMPLES / f"{run}_{track_set}_{colors}.npz",
                        **{f"{t}/{k}": v for t, d in examples.items() for k, v in d.items()})
    return pd.DataFrame(rows), pd.DataFrame(sanity_rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", default=sorted(config.AGENTS))
    p.add_argument("--seeds", type=int, nargs="+", default=config.TRAINING_SEEDS)
    p.add_argument("--colors", nargs="+", default=["normal", "unseen"])
    p.add_argument("--tracks", default="analysis")
    p.add_argument("--recipe", default=config.RECIPE)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    th.backends.cudnn.deterministic, th.backends.cudnn.benchmark = True, False
    th.set_num_threads(config.TORCH_CPU_THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    masks = patch_masks(device=config.DEVICE)
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("forkserver")) as pool:
        for agent in args.agents:
            for seed in args.seeds:
                for colors in args.colors:
                    path = OUT / f"{config.run_name(agent, seed, args.recipe)}_{args.tracks}_{colors}.csv"
                    if path.exists():
                        print(f"skip (done): {path.name}", flush=True)
                        continue
                    sanity = (agent, seed, colors) == ("A", 42, "normal")
                    df, sanity_df = run_one(agent, seed, args.recipe, colors, args.tracks, pool, masks,
                                            load_model(agent, seed, args.recipe, checkpoint=0) if sanity else None)
                    if sanity:
                        sanity_df.to_csv(OUT / "sanity_untrained.csv", index=False)
                    df.to_csv(path, index=False)
                    edge = df["focus_blur_road_edge"].mean()
                    print(f"{path.name}: {len(df)} races, {int(df['moments'].sum())} moments, "
                          f"road-edge focus (blur) {edge:.2f}, max action gap {df['max_action_gap'].max():.1e}", flush=True)


if __name__ == "__main__":
    main()
