"""分析 mc_dump（M6c-OBS 的 6,249 次 MC 决策）：
   1. 所选类 vs 金币（是否"越有钱越花"？—— 诊断储备规则的结构）
   2. 各候选集合下，非 HOLD 的胜出率与分差
   3. MC 与"类头 argmax（=闪电，但闪电不在候选里时等于整回合 HOLD）"的差异率
"""
from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

DUMP = REPO / "training_history" / "vprior" / "mc_dump_obs" / "dump.jsonl"

BINS = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180), (180, 300), (300, 10**9)]


def main() -> int:
    rows = []
    for line in DUMP.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"决策数 = {len(rows)}")

    # 1) 金币 vs 是否出手（出手 = 选 16 或 0；HOLD = 23）
    print("\n[1] 按我方金币分箱：出手（选16/0）占比")
    for lo, hi in BINS:
        sel = [r for r in rows if lo <= r["coins"][r["player"]] < hi]
        if not sel:
            continue
        act = sum(1 for r in sel if r["chosen"] != 23)
        c16 = sum(1 for r in sel if r["chosen"] == 16)
        c0 = sum(1 for r in sel if r["chosen"] == 0)
        print(f"  coins[{lo:>4},{hi if hi < 10**8 else 'inf':>4}) n={len(sel):>5}  "
              f"出手 {act/len(sel):6.1%}  (类16 {c16/len(sel):5.1%}, 类0 {c0/len(sel):5.1%})")

    # 2) 候选集合
    print("\n[2] 候选集合 → 非 HOLD 胜出率 / 中位分差")
    by = collections.defaultdict(list)
    for r in rows:
        by[tuple(r["cands"])].append(r)
    for cands, sel in sorted(by.items(), key=lambda kv: -len(kv[1]))[:5]:
        act = sum(1 for r in sel if r["chosen"] != 23)
        gaps = []
        for r in sel:
            v = sorted(r["scores"].values(), reverse=True)
            if len(v) >= 2:
                gaps.append(v[0] - v[1])
        med = statistics.median(gaps) if gaps else float("nan")
        print(f"  {str(cands):>14} n={len(sel):>5}  非HOLD {act/len(sel):6.1%}  中位分差 {med:.4f}")

    # 3) 金币与"出手"的相关（点二列相关近似：用分箱均值差）
    actx = [r["coins"][r["player"]] for r in rows if r["chosen"] != 23]
    holdx = [r["coins"][r["player"]] for r in rows if r["chosen"] == 23]
    if actx and holdx:
        print(f"\n[3] 出手时的中位金币 = {statistics.median(actx):.0f}（n={len(actx)}）；"
              f"HOLD 时 = {statistics.median(holdx):.0f}（n={len(holdx)}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
