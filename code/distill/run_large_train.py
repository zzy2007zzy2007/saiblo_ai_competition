"""Wrapper to run train_large_balanced.py with forced stdout flushing."""
import sys
import subprocess
from pathlib import Path

log_path = Path("distill_data_v2/train_large.log")
with open(log_path, "w", buffering=1) as log:
    proc = subprocess.Popen(
        [sys.executable, "-u", "code/distill/train_large_balanced.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env={"PYTHONUNBUFFERED": "1", **{k: v for k, v in dict(environ).items() if k != "PYTHONUNBUFFERED"}}
    )
    for line in iter(proc.stdout.readline, b''):
        text = line.decode('utf-8', errors='replace')
        print(text, end='', flush=True)
        log.write(text)
        log.flush()
    proc.wait()
