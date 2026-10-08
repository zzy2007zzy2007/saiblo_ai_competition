"""按局把"churn 强度"与胜负挂钩（NL-R 失败归因用，CPU 很轻）。

对每个 tag：逐局算我方 `建塔/降级/净塔/相邻建降翻转`，然后比较**胜局 vs 负局**的中位数与 Cliff's d。
用法: python code/test_match/churn_vs_win.py <tag> [<tag> ...]
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

OPLINE = re.compile(r"\[round (\d+)\] (p[01])\((\S+?)\) 原文='(.*?)' -> \[(.*?)\]", re.S)
OPNAME = re.compile(r"OperationType\.(\w+): (\d+)>")


def cliff_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (len(a) * len(b))


def per_game(tag: str):
    logdir = REPO / "match_results" / "ladder_logs" / tag
    rows = []
    for f in sorted(logdir.glob("*.log")) if logdir.exists() else []:
        txt = f.read_text(encoding="utf-8", errors="replace")
        line = next((s.strip() for s in txt.splitlines() if s.strip().startswith("RESULT ")), None)
        if not line:
            continue
        kv = dict(re.findall(r"(\w+)=(\S+)", line))
        rev = f.stem.endswith("r")
        us_is_p0 = not rev
        # 镜像 tag：两侧同名 ⇒ 按边读；非镜像沿用 token 约定
        want = "p0" if us_is_p0 else "p1"
        if kv.get("winner_side") in ("p0", "p1"):
            us_won = kv["winner_side"] == want
        else:
            continue
        b = d = flips = 0
        seq = []
        for mm in OPLINE.finditer(txt):
            if int(mm.group(2)[1]) != int(want[1]):
                continue
            names = [nm for nm, _ in OPNAME.findall(mm.group(5))]
            hb, hd = "BUILD_TOWER" in names, "DOWNGRADE_TOWER" in names
            b += names.count("BUILD_TOWER")
            d += names.count("DOWNGRADE_TOWER")
            seq.append((hb, hd))
        for i in range(1, len(seq)):
            if (seq[i - 1][0] and seq[i][1]) or (seq[i - 1][1] and seq[i][0]):
                flips += 1
        rows.append((us_won, b, d, b - d, flips))
    return rows


def main() -> int:
    for tag in sys.argv[1:]:
        rows = per_game(tag)
        w = [r for r in rows if r[0]]
        l = [r for r in rows if not r[0]]
        print(f"{tag}: {len(rows)} 局（胜 {len(w)} / 负 {len(l)}）")
        for i, name in ((1, "建塔"), (2, "降级"), (3, "净塔"), (4, "建降翻转")):
            a = [r[i] for r in w]
            bb = [r[i] for r in l]
            if not a or not bb:
                continue
            print(f"   {name:6s} 胜局中位 {statistics.median(a):6.1f} | 负局中位 {statistics.median(bb):6.1f} "
                  f"| Cliff d = {cliff_d(a, bb):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
