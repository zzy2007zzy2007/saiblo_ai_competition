"""Batch-evaluate top1 of each untested checkpoint in a training directory.

Usage:
    python code/test_match/batch_eval.py training_history/20260630_182516

Runs 50 games with 4 workers per checkpoint, records results to eval_results.md.
Skips checkpoints already recorded in the results file.
"""
from __future__ import annotations
import sys, subprocess, re, time
from pathlib import Path

_RESULTS_FILE = Path(__file__).resolve().parents[2] / "docs" / "eval_results.md"


def _get_tested_gens(results_path: Path, run_dir: str) -> set[int]:
    """Return set of generation numbers already recorded for this run."""
    if not results_path.exists():
        return set()
    text = results_path.read_text(encoding="utf-8")
    # Find the section for this run
    # Pattern: "## 训练 X：`run_dir`" then look for Gen numbers in the table
    in_section = False
    tested: set[int] = set()
    for line in text.splitlines():
        if line.startswith("## ") and f"`{run_dir}`" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break  # next training run
            # Match table rows: | gen | type | games | rate | ...
            m = re.match(r"\|\s*(\d+)\s*\|", line)
            if m:
                tested.add(int(m.group(1)))
    return tested


def _find_latest_gen(run_dir: Path) -> int | None:
    gens = []
    for f in run_dir.glob("gen_*.pt"):
        m = re.match(r"gen_(\d+)\.pt", f.name)
        if m:
            gens.append(int(m.group(1)))
    return max(gens) if gens else None


def _record_result(run_dir: str, gen: int, label: str, games: int, workers: int,
                   wins: int, losses: int, draws: int,
                   w1: int, n1: int, w2: int, n2: int,
                   hp_us_avg: float, hp_opp_avg: float,
                   elapsed: float) -> None:
    """Append a row to the training run's table in eval_results.md."""
    results_path = _RESULTS_FILE
