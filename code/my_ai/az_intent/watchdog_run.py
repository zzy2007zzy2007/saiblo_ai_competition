"""Overnight watchdog for the intent-space AlphaZero training run.

Runs the training command as a subprocess and:
  - on crash (non-zero exit): restart with --resume <checkpoint>
  - on hang (no log output for MAX_STALE seconds): kill + restart with --resume
Gives up after MAX_RESTARTS consecutive problems.

Usage:
    python code/my_ai/az_intent/watchdog_run.py <log_path> <checkpoint> <train_cmd...>
The training subprocess stdout/stderr append to <log_path>; the watchdog's own
messages go to its stdout (the background task output).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LOG_PATH = Path(sys.argv[1])
CHECKPOINT = sys.argv[2]
TRAIN_CMD = list(sys.argv[3:])

POLL_S = 60
MAX_STALE_S = 25 * 60  # no log update for this long -> assume hung
MAX_RESTARTS = 5


def start(cmd: list[str]) -> subprocess.Popen:
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def main() -> None:
    proc = start(TRAIN_CMD)
    last_update = time.time()
    restarts = 0
    while True:
        time.sleep(POLL_S)
        if LOG_PATH.exists():
            last_update = max(last_update, LOG_PATH.stat().st_mtime)
        if proc.poll() is not None:
            code = proc.returncode
            print(f"[watchdog] training exited code={code}", flush=True)
            if code == 0:
                break
            if restarts >= MAX_RESTARTS:
                print("[watchdog] too many restarts, giving up", flush=True)
                break
            restarts += 1
            cmd = TRAIN_CMD + ["--resume", CHECKPOINT]
            print(f"[watchdog] crash -> restart #{restarts} with --resume", flush=True)
            proc = start(cmd)
            last_update = time.time()
            continue
        if time.time() - last_update > MAX_STALE_S:
            print("[watchdog] no log update for 25min, killing & resuming", flush=True)
            proc.kill()
            proc.wait()
            if restarts >= MAX_RESTARTS:
                print("[watchdog] too many restarts, giving up", flush=True)
                break
            restarts += 1
            cmd = TRAIN_CMD + ["--resume", CHECKPOINT]
            print(f"[watchdog] stall -> restart #{restarts} with --resume", flush=True)
            proc = start(cmd)
            last_update = time.time()
    print("[watchdog] done", flush=True)


if __name__ == "__main__":
    main()
