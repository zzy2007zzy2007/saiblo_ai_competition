"""核对内部 A/B：A 侧先/后手拆分。

**权威口径**：直接用 runner（`ab_search_modes.py`）自己打印的行
    [seed 507] flat=P1 score=0.0 rc=0 rounds=338 hp=(17, 0) (16.8 min)
其中 `flat=` 就是 A 侧所在槽位、`score=` 是 A 侧得分（1=胜 0=负）。
这样不依赖任何 env/正则猜测。
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

out_log = Path(sys.argv[1] if len(sys.argv) > 1 else
               "training_history/runs/20261022_100436_ab_upg1_vs_none/output.log")
txt = out_log.read_text(encoding="utf-8", errors="replace")
rows = re.findall(r"\[seed (\d+)\] flat=(P[01]) score=([\d.]+) rc=(-?\d+) rounds=(\d+) hp=\((\d+), (\d+)\)", txt)
print(f"{out_log.name}: 完成 {len(rows)} 局")
split = collections.Counter()
hp_sum = collections.Counter()
for seed, side, score, rc, rounds, h0, h1 in rows:
    won = float(score) >= 1.0
    split[(side, "W" if won else "L")] += 1
    mine, theirs = (int(h0), int(h1)) if side == "P0" else (int(h1), int(h0))
    hp_sum[side] += mine - theirs
for side in ("P0", "P1"):
    w, l = split[(side, "W")], split[(side, "L")]
    n = w + l
    avg = hp_sum[side] / max(1, n)
    print(f"  A 在 {side}: {w}W-{l}L = {100.0*w/max(1,n):5.1f}%（n={n:>2}）  平均血量差 {avg:+.1f}")
tw = sum(v for k, v in split.items() if k[1] == "W")
tl = sum(v for k, v in split.items() if k[1] == "L")
n = tw + tl
print(f"  合计 {tw}W-{tl}L = {100.0*tw/max(1,n):5.1f}%（n={n}）  平均血量差 {(hp_sum['P0']+hp_sum['P1'])/max(1,n):+.1f}")
