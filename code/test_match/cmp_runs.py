"""对比两个 run 在**同一 seed** 上是否逐局复现（确定性检验）。

用法:
  python -u code/test_match/cmp_runs.py M6b_mcq_128 M6b_re128 [--seeds 7-70]
输出：每 seed 的 (winner, base_hp, rounds) 是否相同；总复现局数。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

RESULT = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+) winner_side=(\S+) "
                    r"verdict=(\S+) engine_winner=(\S+) rounds=(\d+) terminal=(\S+) "
                    r"base_hp=(\d+),(\d+)")


def load(tag: str):
    d = REPO / "match_results" / "ladder_logs" / tag
    out = {}
    for f in sorted(d.glob("*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RESULT.search(txt)
        if m:
            out[f.stem] = m.groups()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    args = ap.parse_args()
    a, b = load(args.tag_a), load(args.tag_b)
    common = sorted(set(a) & set(b), key=lambda k: (int(re.sub(r"\D", "", k)), k))
    same = diff = 0
    shown = 0
    for k in common:
        ga, gb = a[k], b[k]
        # 只比 (winner, rounds, base_hp)
        ka = (ga[3], ga[7], ga[9], ga[10])
        kb = (gb[3], gb[7], gb[9], gb[10])
        if ka == kb:
            same += 1
        else:
            diff += 1
            if shown < 6:
                print(f"  差异 {k}: {args.tag_a} winner={ka[0]} rounds={ka[1]} hp={ka[2]},{ka[3]} | "
                      f"{args.tag_b} winner={kb[0]} rounds={kb[1]} hp={kb[2]},{kb[3]}")
                shown += 1
    print(f"公共局 = {len(common)}  逐局相同 = {same}  不同 = {diff}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
