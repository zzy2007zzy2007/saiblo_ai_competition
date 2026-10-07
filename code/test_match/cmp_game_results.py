"""逐局对比两个 tag 的判词（用于零漂移回归：复现的局结果必须与当初**完全一致**）。

用法: python code/test_match/cmp_game_results.py <tagA> <tagB> [seeds=7-70]
输出：一致局数 / 不一致明细（seed、A 的胜者、B 的胜者）。
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

RES = re.compile(r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\S+)")


def load(tag: str, seeds: list[int]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in seeds:
        for suf in ("", "r"):
            p = REPO / "match_results" / "ladder_logs" / tag / f"{s}{suf}.log"
            if not p.exists():
                continue
            m = RES.search(p.read_text(encoding="utf-8", errors="replace"))
            if m:
                out[f"{s}{suf}"] = m.group(4)
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__); return 2
    a, b = sys.argv[1], sys.argv[2]
    spec = sys.argv[3] if len(sys.argv) > 3 else "7-70"
    lo, hi = spec.split("-")
    seeds = list(range(int(lo), int(hi) + 1))
    A, B = load(a, seeds), load(b, seeds)
    common = sorted(set(A) & set(B), key=lambda k: (int(k.rstrip("r")), k.endswith("r")))
    same = [k for k in common if A[k] == B[k]]
    diff = [k for k in common if A[k] != B[k]]
    print(f"{a}（{len(A)} 局） vs {b}（{len(B)} 局）；共同 {len(common)} 局")
    print(f"  判词**完全一致** = {len(same)} / {len(common)}"
          f"{'  ⇒ 零漂移 ✓' if not diff else '  ⇒ **有漂移**'}")
    for k in diff[:12]:
        print(f"    不一致: seed {k}:  A={A[k]}  B={B[k]}")
    if len(A) != len(B):
        only_a = sorted(set(A) - set(B)); only_b = sorted(set(B) - set(A))
        print(f"  只在一侧: A 独有 {only_a[:8]}  B 独有 {only_b[:8]}")
    return 0 if not diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
