"""A/B 目录的血量差统计 + 每局耗时对比。

用法：
  python code/test_match/hp_margin_ab.py <ab目录> [--a-side even|odd]
A 侧 = `--flat-env` 那一侧；在 ab_search_modes.py 里：seed 偶数 → A 侧执 P0，奇数 → P1。
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

_spec = importlib.util.spec_from_file_location("_tmp_ladder", REPO / "_tmp_ladder.py")
_ladder = importlib.util.module_from_spec(_spec)
sys.modules["_tmp_ladder"] = _ladder
_spec.loader.exec_module(_ladder)

out_dir = REPO / sys.argv[1]
rows = []
for log in sorted(out_dir.glob("seed*.log")):
    try:
        seed = int(log.stem.replace("seed", ""))
    except ValueError:
        continue
    parsed = _ladder.read_game(log, str(seed))
    if parsed.get("hp") is None or parsed.get("winner_side") is None:
        continue
    a_is_p0 = (seed % 2 == 0)
    hp = parsed["hp"]
    ours, theirs = (hp[0], hp[1]) if a_is_p0 else (hp[1], hp[0])
    rows.append((seed, a_is_p0, ours, theirs, ours - theirs, parsed.get("rounds"),
                 str(parsed.get("winner_side")).lower()))

if not rows:
    print("无可用对局")
    raise SystemExit(0)

margins = sorted(r[4] for r in rows)
n = len(margins)


def pct(p):
    return margins[min(n - 1, int(p * n))]


wins = sum(1 for r in rows if r[4] > 0)
losses = sum(1 for r in rows if r[4] < 0)
draws = sum(1 for r in rows if r[4] == 0)

print(f"[{out_dir.name}] {n} 局  A 侧血量领先 {wins} / 落后 {losses} / 持平 {draws}")
print(f"\n血量差（A 侧 − B 侧）：")
print(f"  均值 {statistics.mean(margins):+7.2f}   中位 {statistics.median(margins):+7.2f}")
print(f"  p5/p25/p75/p95 = {pct(0.05):+d} / {pct(0.25):+d} / {pct(0.75):+d} / {pct(0.95):+d}")
print(f"  最小 {margins[0]:+d}   最大 {margins[-1]:+d}")
print(f"  平均 |差| = {statistics.mean([abs(m) for m in margins]):.2f}")
se = statistics.stdev(margins) / (n ** 0.5)
print(f"  均值 SE = {se:.2f}  ⇒ 95%CI ≈ [{statistics.mean(margins)-2*se:+.1f}, {statistics.mean(margins)+2*se:+.1f}]")
rounds = [int(r[5]) for r in rows if r[5]]
if rounds:
    print(f"  平均回合 {statistics.mean(rounds):.0f}")
