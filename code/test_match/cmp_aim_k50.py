"""对比 A1（terminal 价值头）与 VK50b（rel+kgeo tau=50 头）在**同一批 seed** 上的机制读数：
   * 我方闪电次数
   * 我方闪电落点到**敌方基地**的切比雪夫距离（越小 = 越贴脸）
   * 我方建塔/升塔数、出招回合数
   * 终局 base_hp 分布
用于回答"kgeo50 改变了什么机制"（不跑对局，只读已有日志）。

用法:
  python -u code/test_match/cmp_aim_k50.py
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

BASE = {0: (16, 9), 1: (2, 9)}          # 敌方基地坐标（我方为 player 时）
OP_LIGHTNING = "USE_LIGHTNING_STORM"
OP_BUILD = "BUILD_TOWER"
OP_UPGRADE = "UPGRADE_TOWER"
RESULT = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) winner_side=(\S+) "
                    r"verdict=(\S+) engine_winner=(\S+) rounds=(\d+) terminal=(\S+) "
                    r"base_hp=(\d+),(\d+) coins=(\d+),(\d+)")


def parse(tag: str):
    d = REPO / "match_results" / "ladder_logs" / tag
    out = []
    for f in sorted(d.glob("*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RESULT.search(txt)
        if not m:
            continue
        seed = int(m.group(1))
        us_is_p0 = (m.group(2) == "az_bridge_ai")
        us = 0 if us_is_p0 else 1
        hp = (int(m.group(10)), int(m.group(11)))
        win = (m.group(4) == "az_bridge_ai")
        ops = {"L": 0, "B": 0, "U": 0}
        dists = []
        rounds = 0
        for mm in re.finditer(r"\[round (\d+)\] (p\d)\(az_bridge_ai\) 原文='[^']*' -> \[(.*?)\]", txt, re.S):
            rounds = max(rounds, int(mm.group(1)))
            for op in re.finditer(r"OperationType\.(\w+): \d+>, arg0=(-?\d+), arg1=(-?\d+)", mm.group(3)):
                kind, a0, a1 = op.group(1), op.group(2), op.group(3)
                if kind == OP_LIGHTNING:
                    ops["L"] += 1
                    if a0 is not None and a1 is not None:
                        bx, by = BASE[us]
                        dists.append(max(abs(int(a0) - bx), abs(int(a1) - by)))
                elif kind == OP_BUILD:
                    ops["B"] += 1
                elif kind == OP_UPGRADE:
                    ops["U"] += 1
        out.append({"seed": seed, "us": us, "win": win, "hp": hp, "ops": ops,
                    "dists": dists, "rounds": rounds})
    return out


def summarize(tag: str):
    rows = parse(tag)
    if not rows:
        print(f"{tag}: 无日志")
        return None
    n = len(rows)
    L = sum(r["ops"]["L"] for r in rows)
    B = sum(r["ops"]["B"] for r in rows)
    U = sum(r["ops"]["U"] for r in rows)
    all_d = [x for r in rows for x in r["dists"]]
    # 只用"我方基地尚存"的局看 hp（避免终局 0）
    print(f"[{tag}] 局数={n}  胜率={sum(r['win'] for r in rows)/n:.3f}")
    print(f"    闪电 {L/n:.2f}/局  建塔 {B/n:.2f}/局  升塔 {U/n:.2f}/局  回合 {statistics.mean(r['rounds'] for r in rows):.0f}")
    if all_d:
        print(f"    闪电落点→敌基地距离: median={statistics.median(all_d):.1f} mean={statistics.mean(all_d):.2f} "
              f"n={len(all_d)}  直击(≤1) 占比={sum(1 for x in all_d if x <= 1)/len(all_d):.1%}")
    return rows


def main() -> int:
    a = summarize("C1_a1_128")     # A1 验收（128 局；这里只用它的机制读数）
    b = summarize("VK50b_32")      # kgeo50（32 局）
    if a and b:
        # 同 seed 对比（c1_a1_128 覆盖 7..70；VK50b_32 覆盖 7..22）
        sa = {r["seed"]: r for r in a}
        common = [s for s in sorted(sa) if s <= 22]
        print(f"\n[同 seed {common[0]}..{common[-1]} 对比]")
        for tag, rows in (("A1", a), ("VK50b", b)):
            sel = [r for r in rows if r["seed"] in common]
            d = [x for r in sel for x in r["dists"]]
            print(f"   {tag:6s} n={len(sel)}  胜={sum(r['win'] for r in sel)/max(len(sel),1):.3f}  "
                  f"闪电={sum(r['ops']['L'] for r in sel)/max(len(sel),1):.2f}  "
                  f"建塔={sum(r['ops']['B'] for r in sel)/max(len(sel),1):.2f}  "
                  f"瞄准中位={statistics.median(d) if d else float('nan'):.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
