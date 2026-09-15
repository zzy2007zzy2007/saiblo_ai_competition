"""Paired per-seed comparison of two vs-rule_v4 eval logs (same seeds, same side assignment).

Both eval.py invocations use --seed 0 --games 64 → seeds 0..63 with our_player=s%2,
so the two runs are matched game-by-game.  Reports the paired win/loss flips
(McNemar) instead of comparing two independent percentages.

Usage: python _tmp_paired_rule_v4.py <logA> <labelA> <first_seed_line> <last_seed_line> <logB> <labelB> ...
Simpler: pass two (log, label, start_line, end_line) tuples positionally.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
PAT = re.compile(r"seed=\s*(\d+)\s+(WIN|DRAW|LOSS)")


def load(path: str, start: int, end: int) -> dict[int, str]:
    out: dict[int, str] = {}
    lines = (REPO / path).read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in lines[start - 1:end]:
        m = PAT.search(ln)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def main() -> None:
    a_path, a_lab, a_s, a_e, b_path, b_lab, b_s, b_e = sys.argv[1:9]
    A = load(a_path, int(a_s), int(a_e))
    B = load(b_path, int(b_s), int(b_e))
    keys = sorted(set(A) & set(B))
    print(f"A={a_lab}: {len(A)} games; B={b_lab}: {len(B)} games; paired on {len(keys)} seeds")

    score = {"WIN": 1.0, "DRAW": 0.5, "LOSS": 0.0}
    wa = sum(score[A[k]] for k in keys) / len(keys)
    wb = sum(score[B[k]] for k in keys) / len(keys)
    both_a_only = sum(1 for k in keys if A[k] == "WIN" and B[k] != "WIN")
    both_b_only = sum(1 for k in keys if B[k] == "WIN" and A[k] != "WIN")
    agree = sum(1 for k in keys if A[k] == B[k])
    print(f"  A win-rate = {wa:.1%}   B win-rate = {wb:.1%}   diff = {wa - wb:+.1%}")
    print(f"  agreement = {agree}/{len(keys)} ({agree / len(keys):.1%})")
    print(f"  A wins where B did not: {both_a_only}   B wins where A did not: {both_b_only}")

    # exact binomial on the discordant pairs
    n = both_a_only + both_b_only
    if n:
        from math import comb
        p = sum(comb(n, i) for i in range(0, min(both_a_only, both_b_only) + 1)) / 2 ** n * 2
        print(f"  McNemar exact two-sided p = {min(p, 1.0):.4f}  (n_discordant={n})")
    print("  per-seed (A/B):", " ".join(f"{k}:{A[k][0]}{B[k][0]}" for k in keys))


if __name__ == "__main__":
    main()
