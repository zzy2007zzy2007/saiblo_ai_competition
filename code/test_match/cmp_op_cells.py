"""位置级机制读数（第 11 轮，预注册 `docs/prereg_20261011_search_up.md` 的 P1）：

`op 直方图`只能看到"做了什么动作"，**看不到"做在哪里"**（K4 的 op 分布与对照几乎一样、分数却 −25pp）。
本脚本对比两个 tag 在**同一 seed** 上的**逐操作落点**：
  * `cell 一致率`：两份日志里顺序对应的我方操作中，`(op_type, arg0, arg1)` 完全相同的比例；
  * `首次分歧`：第一个不同的操作（回合 + 双方的值）。
⇒ 一致率低 = **决策真的变了**（P1 通过）；一致率高而分数差很多 = 差异在别处（该查别的机制量）。

用法: python code/test_match/cmp_op_cells.py <tagA> <tagB> [seeds=7-38]
"""
from __future__ import annotations

import re
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
OP = re.compile(r"OperationType\.(\w+): (\d+)>, arg0=(-?\d+), arg1=(-?\d+)")


def ours(tag: str, seed: int) -> list[tuple[int, int, int, int]]:
    """我方（az_bridge_ai）在该局的逐操作 (round, op_type, arg0, arg1)。"""
    p = REPO / "match_results" / "ladder_logs" / tag / f"{seed}.log"
    if not p.exists():
        return []
    txt = p.read_text(encoding="utf-8", errors="replace")
    m = RESULT.search(txt)
    if not m:
        return []
    us = 0 if m.group(2) == "az_bridge_ai" else 1
    out = []
    for mm in OPLINE.finditer(txt):
        if int(mm.group(2)[1]) != us:
            continue
        for o in OP.finditer(mm.group(5)):
            out.append((int(mm.group(1)), int(o.group(2)), int(o.group(3)), int(o.group(4))))
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    a, b = sys.argv[1], sys.argv[2]
    spec = sys.argv[3] if len(sys.argv) > 3 else "7-38"
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in spec.split(",") if x.strip()]
    tot = same = 0
    per_seed = []
    for s in seeds:
        oa, ob = ours(a, s), ours(b, s)
        n = min(len(oa), len(ob))
        if n == 0:
            continue
        eq = sum(1 for i in range(n) if oa[i] == ob[i])
        tot += n
        same += eq
        first = next((i for i in range(n) if oa[i] != ob[i]), None)
        per_seed.append((s, n, eq / n, oa[first] if first is not None else None,
                         ob[first] if first is not None else None))
    if tot == 0:
        print("没读到对局（检查 tag 名/seed 范围）")
        return 1
    print(f"{a}  vs  {b}   （{len(per_seed)} 个 seed，逐操作对齐 {tot} 个）")
    print(f"  落点完全一致率 = {same / tot:.3f}   （1.000 = 决策一模一样；越低 = 决策变化越大）")
    for s, n, r, fa, fb in per_seed[:6]:
        print(f"    seed {s}: 一致 {r:.2f}（{n} 个操作） 首次分歧: A={fa} B={fb}")
    print(f"  一致率分布: min={min(r for _, _, r, _, _ in per_seed):.2f} "
          f"median={sorted(r for _, _, r, _, _ in per_seed)[len(per_seed) // 2]:.2f} "
          f"max={max(r for _, _, r, _, _ in per_seed):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
