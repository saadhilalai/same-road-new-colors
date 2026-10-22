"""
Memory guard (director's rule, 2026-10-17): this PC also runs other apps, so my jobs must not starve them.
Every 30 s it reads the PC's available memory (Windows "Available MBytes", which is what Task Manager shows;
WSL's own MemAvailable is the fallback). If it is below config.MIN_FREE_PC_MEMORY_GB twice in a row, it stops one
of my jobs: evaluation/analysis jobs first, otherwise the most recently started training. A stopped training loses
nothing but time: it resumes bit-for-bit from its last checkpoint (exact resumes, see scripts/train.py).
Pausing with SIGSTOP would not help, because a paused process keeps its memory.

    setsid nohup nice -n 10 .venv/bin/python -m scripts.memory_guard > /dev/null 2>&1 &

Events go to logs/memory_guard.log (a status line every 10 minutes, and every STOP).
"""

import os
import signal
import subprocess
import time
from datetime import datetime

import config

LOG = config.PROJECT_ROOT / "logs" / "memory_guard.log"
JOB_PATTERNS = ("scripts.diagnose", "scripts.evaluate_agent", "scripts.inspect_driving", "scripts.compare",
                "scripts.attention", "scripts.make_", "scripts.final_test",
                "scripts.train")  # stop order: analysis jobs first, trainings last


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


def windows_available_mb():
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", r"(Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return float(out.replace(",", "."))
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def wsl_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024


def my_jobs():
    """(kind index, start time, pid, args) of my running jobs, one per process group leader."""
    jobs = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            args = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            if os.getpgid(int(pid)) != int(pid) or not os.path.basename(args.split()[0]).startswith("python"):
                continue   # only the Python job itself (a process-group leader), not shells that mention it
            start = os.stat(f"/proc/{pid}").st_mtime
        except (OSError, ProcessLookupError):
            continue
        for kind, pattern in enumerate(JOB_PATTERNS):
            if f"-m {pattern}" in args:
                jobs.append((kind, -start, int(pid), args.strip()))
    return sorted(jobs)


def main():
    limit_mb = config.MIN_FREE_PC_MEMORY_GB * 1024
    log(f"guard started: limit {limit_mb:.0f} MB available")
    low_count, last_status = 0, 0.0
    while True:
        win = windows_available_mb()
        avail = win if win is not None else wsl_available_mb()
        source = "Windows" if win is not None else "WSL"
        low_count = low_count + 1 if avail < limit_mb else 0
        if time.time() - last_status > 600:
            log(f"status: {source} available {avail:.0f} MB, WSL available {wsl_available_mb():.0f} MB, "
                f"my jobs: {[a.split('-m ')[-1][:60] for _, _, _, a in my_jobs()]}")
            last_status = time.time()
        if low_count >= 2:
            jobs = my_jobs()
            if jobs:
                _, _, pid, args = jobs[0]
                os.killpg(pid, signal.SIGTERM)
                log(f"STOP: {source} available {avail:.0f} MB < {limit_mb:.0f} MB -> stopped '{args.split('-m ')[-1]}' (pid {pid})")
            else:
                log(f"LOW: {source} available {avail:.0f} MB, but none of my jobs is running")
            low_count = 0
            time.sleep(60)   # give the memory time to come back before checking again
        time.sleep(30)


if __name__ == "__main__":
    main()
