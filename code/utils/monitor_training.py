"""Daemon that periodically checks training progress and logs status.

Usage:
    # Monitor existing training directory every 30 minutes
    python code/utils/monitor_training.py --dir training_history_20250630_120000 --interval 30

    # Monitor latest training_history_* directory (auto-detect) every 60 minutes
    python code/utils/monitor_training.py --interval 60

    # Specify which PID to watch (optional; otherwise auto-detects python processes on the log dir)
    python code/utils/monitor_training.py --dir training_history_20250630_120000 --pid 12345

The script runs an infinite loop. Stop it with Ctrl+C.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path


def find_latest_training_dir(root: str = ".") -> Path | None:
    """Find the most recent training_history_* directory."""
    dirs = sorted(Path(root).glob("training_history_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def read_last_n_lines(path: Path, n: int = 10) -> list[str]:
    """Read the last N lines of a file efficiently."""
    if not path.exists():
        return [f"[FILE NOT FOUND: {path}]"]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return [l.rstrip("\n\r") for l in lines[-n:]]
    except Exception as e:
        return [f"[ERROR reading {path.name}: {e}]"]


def read_csv_tail(path: Path, n: int = 3) -> list[dict[str, str]]:
    """Read the last N rows of a CSV file as dicts."""
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-n:]
    except Exception as e:
        return [{"error": str(e)}]


def find_training_pids(log_dir: Path, log_path: Path) -> list[int]:
    """Find python processes that have the log_dir or log_path open.

    Uses `lsof` on macOS/Linux, falls back to `wmic` + tasklist on Windows.
    Returns empty list if detection fails (process may still be running).
    """
    pids: list[int] = []
    try:
        if sys.platform == "win32":
            # Try to find python processes
            import subprocess
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq python3.exe", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, timeout=5,
                )
            # Extract PIDs
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip('"') for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        pids.append(int(parts[1]))
                    except ValueError:
                        pass
        else:
            # macOS/Linux: use lsof to find processes with open handle to log_dir
            import subprocess
            result = subprocess.run(
                ["lsof", "-t", str(log_path)],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                pids = [int(pid) for pid in result.stdout.strip().split("\n") if pid]
    except Exception:
        pass
    return pids


def check_process_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        if sys.platform == "win32":
            import subprocess
