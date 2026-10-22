"""
Put the included trained models (models/) and the recorded final-test actions (results/final_test/actions/) where
every script looks for them (runs/<run name>/checkpoints/ and runs/<run name>/actions/), so the evaluations, the final
test's videos, the charts and the website data can be re-made without training.

    python -m scripts.restore_models
"""

import shutil

import config

MODELS = config.PROJECT_ROOT / "models"
ACTIONS = config.RESULTS_DIR / "final_test" / "actions"


def main():
    for src in sorted(MODELS.glob("*.zip")):
        run, _, checkpoint = src.stem.partition("__")          # e.g. A_seed42_beta__final.zip
        dst = config.RUNS_DIR / run / "checkpoints" / f"model_{checkpoint}.zip"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)
        print(f"{src.name} -> {dst.relative_to(config.PROJECT_ROOT)}")
    for src in sorted(ACTIONS.glob("*.npz")):
        run, _, name = src.name.partition("__")                 # e.g. A_seed42_beta__test_normal.npz
        dst = config.RUNS_DIR / run / "actions" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy(src, dst)
    print(f"recorded test actions: {len(list(ACTIONS.glob('*.npz')))} files restored")


if __name__ == "__main__":
    main()
