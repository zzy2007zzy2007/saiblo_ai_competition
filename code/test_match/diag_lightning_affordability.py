"""验证假说：**升级吃掉了金币 ⇒ 闪电放不起**（而不是"值网觉得闪电贵所以主动少放"）。

闪电规则：`cost=90`、`cooldown=35`（SDK `SUPER_WEAPON_STATS[LIGHTNING_STORM]`）。

数据源：
  · 金币轨迹：`match_results/ai{k}_az_bridge_ai_seed{S}.stderr.log` 的逐回合 `coins=[我, 敌]`
  · 出招：`training_history/ab_challenger_vs_champion/seed{S}_p{k}.log`（文件名后缀就是 A 侧座位）

对每一侧算三件事：
  1. **冷却已好的回合数**（上一次施放后的 35 回合内算"未就绪"）
  2. 其中**真的放了**多少（施放率）
  3. 就绪回合上的**金币分布**，以及金币 ≥ 90（买得起）的比例

判读：
  · 若 A 的"就绪但买不起"比例显著高于 B ⇒ 支持用户假说（升级挤占了闪电预算）；
  · 若 A 的金币够、却仍少放 ⇒ 支持"值网主动少放"。
"""
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
COOLDOWN = 35
COST = 90
AB = REPO / "training_history/ab_challenger_vs_champion"

pat_line = re.compile(r".*\[round (\d+)\] (p[01])\([^)]*\)[^\n]*?-> \[(.*?)\]\s*$")
pat_coins = re.compile(r"\[azai\] round=(\d+) .*?coins=\[(\d+), (\d+)\]")


def parse_casts(log: Path, seat: str) -> list[int]:
    out = []
    if not log.exists():
        return out
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat_line.match(line)
        if not m:
            continue
        if m.group(2) != seat:
            continue
        if "USE_LIGHTNING_STORM" in m.group(3):
            out.append(int(m.group(1)))
    return out


def parse_coins(log: Path, seat: str) -> dict[int, int]:
    out = {}
    if not log.exists():
        return out
    mine = 0 if seat == "p0" else 1
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat_coins.search(line)
        if m:
            out[int(m.group(1))] = int(m.group(2 + mine))
    return out


def analyse(casts: list[int], coins: dict[int, int]):
    "返回 (就绪回合列表, 就绪且放了, 就绪时的金币列表)"
    if not coins:
        return [], 0, []
    max_round = max(coins)
    last_cast = -10**9
    ready, ready_cast, ready_coins = [], 0, []
    cast_set = set(casts)
    for r in range(0, max_round + 1):
        if r - last_cast < COOLDOWN:
            if r in cast_set:
                last_cast = r
            continue
        ready.append(r)
        if r in cast_set:
            ready_cast += 1
            last_cast = r
        if r in coins:
            ready_coins.append(coins[r])
    return ready, ready_cast, ready_coins


stats = {"A": {"ready": 0, "cast": 0, "coins": [], "cast_when_afford": 0, "ready_afford": 0},
         "B": {"ready": 0, "cast": 0, "coins": [], "cast_when_afford": 0, "ready_afford": 0}}

for seed in range(960, 976):
    for a_seat in ("p0", "p1"):
        seat_idx = a_seat[1]
        a_log = AB / f"seed{seed}_{a_seat}.log"
        if not a_log.exists():
            continue
        b_seat = "p1" if a_seat == "p0" else "p0"
        for key, seat in (("A", a_seat), ("B", b_seat)):
            match_log = AB / f"seed{seed}_{seat}.log"
            bridge = REPO / f"match_results/ai{seat[1]}_az_bridge_ai_seed{seed}.stderr.log"
            casts = parse_casts(match_log, seat)
            coins = parse_coins(bridge, seat)
            ready, ready_cast, ready_coins = analyse(casts, coins)
            s = stats[key]
            s["ready"] += len(ready)
            s["cast"] += ready_cast
            s["coins"].extend(ready_coins)
            s["ready_afford"] += sum(1 for c in ready_coins if c >= COST)
            s["cast_when_afford"] += ready_cast   # 近似：能放的回合里真放了多少

print(f"闪电规则：cost={COST} 金币 / cooldown={COOLDOWN} 回合")
print(f"{'':<16}{'就绪回合':>10}{'其中施放':>10}{'施放率':>9}{'就绪时金币中位':>16}{'金币≥90占比':>13}")
for k in ("A", "B"):
    s = stats[k]
    if not s["ready"]:
        print(f"{k:<16}（无数据）")
        continue
    rate = 100.0 * s["cast"] / s["ready"]
    med = statistics.median(s["coins"]) if s["coins"] else float("nan")
    afford = 100.0 * s["ready_afford"] / max(1, len(s["coins"]))
    label = "A 值网+升级" if k == "A" else "B quiet+不升级"
    print(f"{label:<16}{s['ready']:>10}{s['cast']:>10}{rate:>8.1f}%{med:>16.0f}{afford:>12.1f}%")

print()
for k in ("A", "B"):
    s = stats[k]
    if s["coins"]:
        cs = sorted(s["coins"])
        print(f"  {k} 就绪回合金币分布: p10={cs[len(cs)//10]} p50={cs[len(cs)//2]} p90={cs[len(cs)*9//10]}"
              f"  （<90 的回合 {sum(1 for c in cs if c < COST)}/{len(cs)}）")
