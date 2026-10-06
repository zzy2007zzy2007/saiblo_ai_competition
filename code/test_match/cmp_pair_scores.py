"""比较两个 tag 在**同一批 seed** 上的逐对胜负（看"换了评价器之后，决策变了多少、分数变了多少"）。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
for _s in (sys.stdout, sys.stderr):          # ⚠️ Windows 控制台默认 GBK ⇒ 不重设会 UnicodeEncodeError
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
RESULT = re.compile(r"RESULT seed=(\d+) .*?winner=(\S+) winner_side=")


def outcomes(tag: str, seeds: list[int]) -> dict[tuple[int, str], str]:
    out = {}
    for s in seeds:
        for suf in ("", "r"):
            p = REPO / "match_results" / "ladder_logs" / tag / f"{s}{suf}.log"
            if not p.exists():
                continue
            m = RESULT.search(p.read_text(encoding="utf-8", errors="replace"))
            if m:
                out[(s, suf or "n")] = m.group(2)
    return out


def pair_score(tag: str, seeds: list[int]) -> tuple[float, int, int, int, list[int]]:
    """→ (配对胜率, 2-0 数, 1-1 数, 0-2 数, 得分=0/1/2 的列表)"""
    o = outcomes(tag, seeds)
    scores, w20, d11, l02 = [], 0, 0, 0
    us = "az_bridge_ai"
    for s in seeds:
        a = o.get((s, "n")); b = o.get((s, "r"))
        if not a or not b:
            continue
        k = (a == us) + (b == us)
        scores.append(k)
        if k == 2:
            w20 += 1
        elif k == 1:
            d11 += 1
        else:
            l02 += 1
    return (sum(scores) / (2 * len(scores)) if scores else float("nan"),
            w20, d11, l02, scores)


def main() -> int:
    tag_a, tag_b = sys.argv[1], sys.argv[2]
    seeds = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else list(range(7, 39))
    pa, w1, d1, l1, sa = pair_score(tag_a, seeds)
    pb, w2, d2, l2, sb = pair_score(tag_b, seeds)
    print(f"{tag_a}: p̂={pa:.4f}  ({w1}/{d1}/{l1})  n对={len(sa)}")
    print(f"{tag_b}: p̂={pb:.4f}  ({w2}/{d2}/{l2})  n对={len(sb)}")
    print(f"差 = {pb - pa:+.4f}")
    flips = sum(1 for x, y in zip(sa, sb) if x != y)
    print(f"逐对**翻转**数 = {flips} / {len(sa)}（{100*flips/max(1,len(sa)):.0f}% 的对变了）")
    print("逐对细节（seed: A→B）:")
    for s, x, y in zip(seeds, sa, sb):
        if x != y:
            print(f"   seed {s}: {x} → {y}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

