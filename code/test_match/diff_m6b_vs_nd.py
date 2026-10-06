"""对比 M6b_mcq_128 与 M6bND_128 在**同一 seed** 上的逐决策分歧（只读日志）。

目的：0.6641 vs 0.4297 差 23pp，但机制计数几乎一样 ⇒ 找出**第一个分歧点**，
判断是"类 16 被去掉"导致的行为变化，还是别的。
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

OPLINE = re.compile(r"\[round (\d+)\] (p\d)\(az_bridge_ai\) 原文='([^']*)' -> \[(.*?)\]")


def decisions(tag: str, name: str):
    f = REPO / "match_results" / "ladder_logs" / tag / name
    if not f.exists():
        return None
    txt = f.read_text(encoding="utf-8", errors="replace")
    out = []
    for m in OPLINE.finditer(txt):
        out.append((int(m.group(1)), m.group(3).replace("\n", " "), m.group(4)))
    return out


def main() -> int:
    seeds = [7, 8, 9, 10, 11, 12]
    for s in seeds:
        for suffix in ("", "r"):
            name = f"{s}{suffix}.log"
            a = decisions("M6b_mcq_128", name)
            b = decisions("M6bND_128", name)
            if a is None or b is None:
                print(f"{name}: 缺日志")
                continue
            n = min(len(a), len(b))
            first = None
            for i in range(n):
                if a[i][1] != b[i][1] or a[i][2] != b[i][2]:
                    first = i
                    break
            same = (len(a) == len(b)) and all(a[i][1:] == b[i][1:] for i in range(n))
            print(f"== {name}: M6b {len(a)} 次决策 / M6bND {len(b)} 次；完全一致={same}")
            if first is not None:
                print(f"   首个分歧 @第 {first} 次决策 (round {a[first][0]}):")
                for tag, seq in (("M6b  ", a), ("M6bND", b)):
                    if first < len(seq):
                        print(f"     {tag} round={seq[first][0]:>3} 原文='{seq[first][1]}' -> {seq[first][2][:90]}")
                # 分歧点**之前**两边都用过哪些 op（看 M6b 是否更早出现降级）
                for tag, seq in (("M6b  ", a), ("M6bND", b)):
                    pre = [x[2] for x in seq[:first]]
                    cnt = {}
                    for t in pre:
                        for mm in re.finditer(r"OperationType\.(\w+)", t):
                            cnt[mm.group(1)] = cnt.get(mm.group(1), 0) + 1
                    print(f"     {tag} 分歧前 ops: {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
