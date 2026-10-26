"""循环赛统计：读各目录的 seed*.log，按 RESULT 行统计胜率/血量差/座位拆分。"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
RES = re.compile(
    r"RESULT seed=(\d+) p0=(\S+) p1=(\S+) winner=(\w+) winner_side=(\w+) verdict=(\w+).*?"
    r"base_hp=(\d+),(\d+)"
)


def stats(dirname: str, a_label: str) -> None:
    d = REPO / "training_history" / dirname
    if not d.exists():
        print(f"  [{dirname}] 不存在")
        return
    rows = []
    for f in sorted(d.glob("seed*_p*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = RES.search(txt)
        if not m:
            continue
        verdict = m.group(6)
        if verdict != "engine":
            continue
        h0, h1 = int(m.group(7)), int(m.group(8))
        side = "p0" if f.stem.endswith("_p0") else "p1"
        mine, theirs = (h0, h1) if side == "p0" else (h1, h0)
        rows.append((mine, theirs, side))
    if not rows:
        print(f"  [{dirname}] 无有效局")
        return
    ms = [r[0] - r[1] for r in rows]
    w = sum(1 for x in ms if x > 0)
    n = len(rows)
    mu = statistics.mean(ms)
    se = statistics.stdev(ms) / n ** 0.5 if n > 1 else 0
    print(f"  [{dirname}]  A={a_label}  {w}W-{n-w}L = {100*w/n:.1f}%   "
          f"血量差 {mu:+.2f} 95%CI [{mu-1.96*se:+.1f},{mu+1.96*se:+.1f}]  (n={n})")
    for s in ("p0", "p1"):
        sub = [r for r in rows if r[2] == s]
        if sub:
            ms2 = [r[0] - r[1] for r in sub]
            w2 = sum(1 for x in ms2 if x > 0)
            print(f"       A@{s}: {w2}W-{len(sub)-w2}L = {100*w2/len(sub):.0f}%   血量差 {statistics.mean(ms2):+.1f}")


def main() -> int:
    stats("rr_f10_vs_hybrid", "f10")
    stats("rr_f10_vs_pz", "f10")
    stats("rr_pz_vs_hybrid", "prior_z")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
