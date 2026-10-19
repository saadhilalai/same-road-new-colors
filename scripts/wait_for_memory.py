"""
Waits until the PC has memory to spare again, then exits (rule written in PROGRESS.md before first use, 2026-10-18).
After the memory guard stops a job, the job is restarted only once Windows' available memory has stayed at or above
config.RESUME_MIN_FREE_PC_MEMORY_GB for config.RESUME_STEADY_MINUTES without a dip: the 3 GB floor plus the ~2 GB
one job costs on the Windows side, plus a margin, so the restart does not immediately trip the guard again.

    .venv/bin/python -m scripts.wait_for_memory && <restart the job>

A job's FIRST start (not a restart after a memory stop) only needs the memory steady for
config.FIRST_START_STEADY_MINUTES (--first-start), added 2026-10-19: a 10-minute wait before each of about
15 jobs in a row would add over 2 hours.
"""

import argparse
import time

import config
from scripts.memory_guard import log, windows_available_mb, wsl_available_mb

CHECK_EVERY_S = 30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first-start", action="store_true")
    minutes = config.FIRST_START_STEADY_MINUTES if p.parse_args().first_start else config.RESUME_STEADY_MINUTES
    need_mb = config.RESUME_MIN_FREE_PC_MEMORY_GB * 1024
    steady_needed = max(1, minutes * 60 // CHECK_EVERY_S)
    log(f"WAIT: {'start' if minutes != config.RESUME_STEADY_MINUTES else 'restart'} needs >= {need_mb:.0f} MB "
        f"available for {minutes} min")
    steady, last_status = 0, time.time()
    while True:
        win = windows_available_mb()
        avail = win if win is not None else wsl_available_mb()
        steady = steady + 1 if avail >= need_mb else 0
        if steady >= steady_needed:
            log(f"RECOVERED: {avail:.0f} MB available, steady for {minutes} min -> starting")
            return
        if time.time() - last_status > 1800:
            log(f"WAIT: still waiting, {avail:.0f} MB available now")
            last_status = time.time()
        time.sleep(CHECK_EVERY_S)


if __name__ == "__main__":
    main()
