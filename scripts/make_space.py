"""
Step 8: pack the website into a folder that runs as a free Hugging Face "static" Space, and optionally upload it.
The site needs no server: the page (web/), the recorded test races it replays (web/data/, from
scripts.export_site_data), the videos (web/media/), the S4D marks and the Geist fonts.

    python -m scripts.make_space                                   # build build/hf_space/ only
    python -m scripts.make_space --push USERNAME/same-road-new-colors   # build, then upload (hf auth login first)
"""

import argparse
import shutil

import config

OUT = config.PROJECT_ROOT / "build" / "hf_space"
WEB = config.PROJECT_ROOT / "web"
VIDEOS = {"3_a_vs_b_unseen_colors.mp4": "a_vs_b_unseen_colors.mp4", "4_split_screen_what_the_ai_sees.mp4": "split_screen.mp4"}

SPACE_CARD = """\
---
title: Same Road, New Colors
colorFrom: gray
colorTo: purple
sdk: static
app_file: index.html
pinned: false
short_description: A pixel-only race-car AI vs new colors, 1,200 test races
---

# Same Road, New Colors

Two race-car AIs taught themselves to drive from camera pixels only (Gymnasium CarRacing), with exactly the same
practice. One only ever saw the normal colors; the other saw a new random color scheme in every training race.
On 100 hidden test tracks in colors neither had seen, the one that practised with random colors scored 581 against
92 (tested once; paired test, p < 0.001), at a small price on normal colors (739 vs 807). Every race on the page is
a replay of a recorded test run, checked against the recorded result.

Designed & built by Saad Hilal · S4D. Code, test plan and lab notebook:
https://github.com/saadhilalai/same-road-new-colors

Simulator: Gymnasium CarRacing-v3 (Box2D). Learning: Stable-Baselines3 PPO with a Beta policy (Petrazzini &
Antonelo 2021 settings). Typeface: Geist and Geist Mono by Vercel (SIL Open Font License, see fonts/OFL.txt).
Motion: GSAP 3.14.2 with ScrollTrigger (GreenSock standard "no charge" license).
"""


def check_inputs():
    tracks = sorted((WEB / "data" / "tracks").glob("*.json"))
    missing = [p for p in (WEB / "index.html", WEB / "data" / "site.json", WEB / "media" / "poster.jpg",
                           *(config.RESULTS_DIR / "final_test" / "videos" / v for v in VIDEOS)) if not p.exists()]
    if len(tracks) != len(config.TEST_TRACKS) or missing:
        raise SystemExit(f"Missing site data ({len(tracks)}/{len(config.TEST_TRACKS)} tracks, missing {missing}). "
                         "Run: python -m scripts.export_site_data")


def build():
    check_inputs()
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(WEB, OUT, ignore=shutil.ignore_patterns(".DS_Store", "*.map"))
    for src, dst in VIDEOS.items():                       # the site's two clips, from the final-test videos
        shutil.copy(config.RESULTS_DIR / "final_test" / "videos" / src, OUT / "media" / dst)
    (OUT / "README.md").write_text(SPACE_CARD)
    files = [p for p in OUT.rglob("*") if p.is_file()]
    print(f"Built {OUT.relative_to(config.PROJECT_ROOT)}/ ({len(files)} files, "
          f"{sum(p.stat().st_size for p in files) / 1e6:.1f} MB)")


def push(repo_id):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(folder_path=str(OUT), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy Same Road, New Colors", delete_patterns="*")
    user, name = repo_id.split("/")
    print(f"Uploaded. Space page: https://huggingface.co/spaces/{repo_id}")
    print(f"Live site: https://{user.lower()}-{name.lower().replace('_', '-')}.static.hf.space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", metavar="USERNAME/SPACE", help="upload to this Hugging Face Space")
    args = parser.parse_args()
    build()
    if args.push:
        push(args.push)
