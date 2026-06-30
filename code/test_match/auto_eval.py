"""Auto-evaluate the latest checkpoint in a training directory.

Usage:
    python code/test_match/auto_eval.py training_history/20260630_182516

Loops forever: watches for new .pt files, evaluates each once with
50 games / 4 workers (mean + top1), logs results, then sleeps until
a newer checkpoint appears.
"""
from __future__ import annotations
import sys, time, subprocess, re, argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

EVAL_SCRIPT = Path(__file__).resolve().parent / "eval_checkpoint.py"
