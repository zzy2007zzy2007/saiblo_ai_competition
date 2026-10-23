"""汇总所有"随机化(只对手) vs 确定性冠军"的臂：胜率 + 血量差（2026-10-23）。

因为单臂只有 32 局（胜率 CI ±17pp），把 5 条臂合起来看更有意义。
血量差是连续量，灵敏度高得多。
"""
import math
import re
import statistics
from pathlib import Path

ARMS = [
    ("training_history/runs/20261023_152048_ab_rollout_decay_vs_idle_v2", "N=8"),
    ("training_history/ab_decay_s32", "N=32"),
    ("training_history/ab_random_bign_tree", "N=64"),
    ("training_history/ab_random_n128_tree", "N=128"),
    ("training_history/ab_dedupe_n256", "N=256+去重"),
]

all_margins = []
all_wins = []
print("  随机化(只对手) vs 确定性冠军：")
for d, label in ARMS:
    base = Path(d)
    if not base.exists():
        print(f"    {label:<12} 目录缺失")
        continue
    margins, wins, n = [], 0, 0
    for f in sorted(base.glob("seed*_p*.log")):
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"base_hp=(\d+),(\d+)", txt)
        w = re.search(r"winner_side=(p[01])", txt)
        if not m or not w:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        side0 = "p0" if f.stem.endswith("_p0") else "p1"   # 该局 flat(A) 坐的位置
        mine, theirs = (a, b) if side0 == "p0" else (b, a)
        margins.append(mine - theirs)
        wins += 1 if w.group(1) == side0 else 0
        n += 1
    if n:
        mu = statistics.mean(margins)
        se = statistics.stdev(margins) / math.sqrt(n)
        print(f"    {label:<12} {wins}W-{n - wins}L = {100 * wins / n:5.1f}%   "
              f"血量差 {mu:+6.2f} ±{1.96 * se:4.2f}  (n={n})")
        all_margins += margins
        all_wins += [1] * wins + [0] * (n - wins)

n = len(all_wins)
p = sum(all_wins) / n
se_w = math.sqrt(p * (1 - p) / n)
mu = statistics.mean(all_margins)
se_m = statistics.stdev(all_margins) / math.sqrt(len(all_margins))
print(f"\n  **汇总胜率 {sum(all_wins)}W-{n - sum(all_wins)}L = {100 * p:.1f}%**  "
      f"95%CI [{100 * (p - 1.96 * se_w):.1f}%, {100 * (p + 1.96 * se_w):.1f}%]  n={n}")
print(f"  **汇总血量差 {mu:+.2f} hp**  95%CI [{mu - 1.96 * se_m:+.2f}, {mu + 1.96 * se_m:+.2f}]")
verdict_w = "显著" if abs(p - 0.5) > 1.96 * se_w else "不显著"
verdict_m = "显著" if mu - 1.96 * se_m > 0 else "不显著"
print(f"  ⇒ 胜率相对 50%：{verdict_w}；血量差相对 0：{verdict_m}")
