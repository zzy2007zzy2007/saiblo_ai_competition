#!/usr/bin/env python3
"""离线消融：把亚军候选打分里的"价值项"去掉后，每回合 argmax 会不会变。

用 `debug=plans` 的 per-plan 日志（每行一个候选，含 `score`/`score_before_heuristic`/
`mean_terminal_score`/`heuristic`/`rank`）。

对每次决策（round, player, serial）:
  A. argmax(score)                      —— 亚军实际的"最佳候选"
  B. argmax(score - value)              —— 去掉价值项（= heuristic - penalty）
  C. argmax(mean_terminal_score)        —— 只用价值
统计 A 与 B/C 的 argmax 是否一致；不一致 ⇒ 价值确实改变了选择。
注意：这是"同一局面"的反事实，**不是重放**（选择变了局面也会变），所以只是上界/近似。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]


def load_plans(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('{"kind":"plan"'):
                try:
                    rows.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    pass
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="match_results")
    ap.add_argument("--glob", default="ai*_runnerup_seed900.stderr.log")
    a = ap.parse_args()
    d = Path(a.dir)
    if not d.is_absolute():
        d = _REPO / d

    by_dec = defaultdict(list)
    for f in sorted(d.glob(a.glob)):
        for r in load_plans(f):
            by_dec[(int(r.get("round", -1)), int(r.get("player", -1)), int(r.get("serial", -1)))].append(r)
    print(f"[abl] 决策 {len(by_dec)}", flush=True)

    def argmax_key(cands, keyfn):
        best, bestv = None, -1e18
        for c in cands:
            v = keyfn(c)
            if v is None:
                continue
            if v > bestv:
                bestv, best = v, c.get("key")
        return best

    n = 0
    same_no_value = 0
    same_value_only = 0
    rank_changed = 0
    diffs = []
    for key, cands in by_dec.items():
        if len(cands) < 2:
            continue
        n += 1
        a_key = argmax_key(cands, lambda c: c.get("score"))
        b_key = argmax_key(cands, lambda c: (c.get("score", 0.0) - (c.get("mean_terminal_score") or 0.0)))
        c_key = argmax_key(cands, lambda c: c.get("mean_terminal_score"))
        if a_key == b_key:
            same_no_value += 1
        if a_key == c_key:
            same_value_only += 1
        # 亚军实际选的（rank=1）是哪个？
        chosen = next((c for c in cands if c.get("rank") == 1), None)
        if chosen is not None and chosen.get("key") != a_key:
            rank_changed += 1

    print(f"[abl] 多候选决策 {n}", flush=True)
    print(f"[abl] argmax(去价值) == argmax(原score) : {same_no_value}/{n} "
          f"({100.0*same_no_value/max(n,1):.1f}%)  -> 越接近100%说明价值越不影响选择", flush=True)
    print(f"[abl] argmax(只用价值) == argmax(原score): {same_value_only}/{n} "
          f"({100.0*same_value_only/max(n,1):.1f}%)", flush=True)
    print(f"[abl] rank=1 与 argmax(score) 不一致: {rank_changed}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
