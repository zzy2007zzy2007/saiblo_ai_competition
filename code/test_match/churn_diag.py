"""churn 刻画：把"建塔↔降级"的循环量化（Phase 0 / NL-R 用）。

每 tag 输出（我方）：
  * 建 / 降 / 净塔数（建−降）；
  * **同一回合内**同时出现"建塔+降级"的回合数（束内自循环）；
  * **相邻两回合**出现"建→降"或"降→建"的**翻转**次数与占比（跨回合 churn）；
  * 操作密集度：有操作回合数 / 对局回合数。

用法: python code/test_match/churn_diag.py <tag> [<tag> ...]
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+)")
OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPNAME = re.compile(r"OperationType\.(\w+): (\d+)>")


def one(tag: str) -> dict:
    logdir = REPO / "match_results" / "ladder_logs" / tag
    games = 0
    b = d = net = 0
    both_same_round = 0
    flips = 0
    act_rounds = 0
    rounds_list = []
    per_game_net = []
    for f in sorted(logdir.glob("*.log")) if logdir.exists() else []:
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RESULT.search(txt)
        if not m:
            continue
        us = 0 if m.group(2) == "az_bridge_ai" else 1
        games += 1
        seq = []          # 我方按回合的 (build?, down?)
        for mm in OPLINE.finditer(txt):
            if int(mm.group(2)[1]) != us:
                continue
            names = [nm for nm, _ in OPNAME.findall(mm.group(5))]
            hb = "BUILD_TOWER" in names
            hd = "DOWNGRADE_TOWER" in names
            if hb or hd:
                act_rounds += 1
                seq.append((hb, hd))
                if hb and hd:
                    both_same_round += 1
                b += sum(1 for x in names if x == "BUILD_TOWER")
                d += sum(1 for x in names if x == "DOWNGRADE_TOWER")
        for i in range(1, len(seq)):
            if (seq[i - 1][0] and seq[i][1]) or (seq[i - 1][1] and seq[i][0]):
                flips += 1
        rr = re.search(r"rounds=(\d+)", txt)
        if rr:
            rounds_list.append(int(rr.group(1)))
        per_game_net.append(sum(1 for x in seq if x[0]) - sum(1 for x in seq if x[1]))
    g = max(games, 1)
    return dict(tag=tag, games=games, b=b / g, d=d / g, net=(b - d) / g,
                both=both_same_round / g, flips=flips / g,
                flip_rate=flips / max(act_rounds, 1), act=act_rounds / g,
                rounds=statistics.mean(rounds_list) if rounds_list else float("nan"),
                net_game=statistics.mean(per_game_net) if per_game_net else float("nan"))


def main() -> int:
    tags = sys.argv[1:]
    if not tags:
        print(__doc__); return 2
    print(f"{'tag':20s} {'局':>4s} {'建/局':>7s} {'降/局':>7s} {'净塔/局':>8s} "
          f"{'同回合建+降/局':>14s} {'相邻翻转/局':>12s} {'翻转率':>8s} {'有操作回合':>10s}")
    for t in tags:
        r = one(t)
        print(f"{r['tag']:20s} {r['games']:4d} {r['b']:7.1f} {r['d']:7.1f} {r['net']:8.1f} "
              f"{r['both']:14.1f} {r['flips']:12.1f} {r['flip_rate']:8.2f} {r['act']:10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
