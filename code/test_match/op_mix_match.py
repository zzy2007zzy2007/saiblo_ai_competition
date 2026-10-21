"""统计一场 ladder 对局（或整批）里双方的出招类型构成。

用法：python code/test_match/op_mix_match.py match_results/ladder_logs/VS_RUNNERUP_cells4
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

log_dir = Path(sys.argv[1])
pat_op = re.compile(r"OperationType\.(\w+)")
pat_side = re.compile(r"p([01])\((\w+)\).*?->")

ours = collections.Counter()
theirs = collections.Counter()
games = 0
for log in sorted(log_dir.glob("*.log")):
    ours_game, theirs_game = collections.Counter(), collections.Counter()
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat_side.search(line)
        if not m:
            continue
        side, name = m.group(1), m.group(2)
        ops = pat_op.findall(line)
        if not ops:
            continue
        (ours_game if name.startswith("az_bridge") else theirs_game).update(ops)
    if ours_game or theirs_game:
        games += 1
        ours.update(ours_game)
        theirs.update(theirs_game)

def show(title, counter):
    total = sum(counter.values())
    print(f"\n=== {title}（{total} 个 op）===")
    for name, count in counter.most_common():
        print(f"  {name:<24} {count:>6}  {100*count/max(total,1):5.1f}%")

print(f"[{log_dir.name}] 统计了 {games} 局")
show("我们", ours)
show("对手", theirs)

if ours and theirs:
    print("\n=== 差异（百分点）===")
    to, tt = sum(ours.values()), sum(theirs.values())
    for name in sorted(set(ours) | set(theirs)):
        a, b = 100 * ours[name] / to, 100 * theirs[name] / tt
        print(f"  {name:<24} 我们 {a:5.1f}%   对手 {b:5.1f}%   差 {a-b:+6.1f}pp")
