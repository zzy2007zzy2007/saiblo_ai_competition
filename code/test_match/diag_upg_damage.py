"""机制诊断：升级到底伤在哪一环？

数据源（全部离线，不占 CPU）：
  · 桥接 stderr：`match_results/ai{0,1}_az_bridge_ai_seed<N>.stderr.log`
    每回合一行 `[azai] round=R ... coins=[我, 敌] towers=N lv=[(x,y,type,lv)...]`
  · 对局日志：`training_history/ab_upg1_vs_none/seed<N>.log`（双方逐回合 op，可数建塔/降级）

A/B 判定：`ab_search_modes.py` 的规则是「seed 偶数 ⇒ A 侧执 P0（即 ai0）」，
          所以 A 侧日志 = `ai0_...seed<K>.stderr.log`（K 偶数）/ `ai1_...`（K 奇数）。

输出三组对比（A=有升级 / B=无升级）：
  1. **金币轨迹**（按回合分段）——验"升级是否让我们更穷"（假说 B/C）
  2. **塔数轨迹** —— 验"升级是否换来了塔的数量劣势"
  3. **建塔/降级次数与"塔消失"次数** —— 验"charm 经济被破坏"（假说 B）
"""
from __future__ import annotations

import collections
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AB = REPO / "training_history/ab_upg1_vs_none"
SEEDS = list(range(500, 564))          # arm ①: 有升级 vs 无升级
BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 260), (260, 400)]


def parse_bridge(path: Path):
    """返回 [(round, coins_mine, towers_mine, lv_list)]。"""
    out = []
    if not path.exists():
        return out
    pat = re.compile(r"\[azai\] round=(\d+) .*?coins=\[(\d+), (\d+)\] towers=(\d+) lv=\[(.*?)\]")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        rnd, c0, c1 = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # ⚠️ 	owers=N 是**全场塔数**（双方相同，不能用）；我方塔在 lv=[(x,y,type,lv), ...]
        lv_s = m.group(5)
        own = lv_s.count('(')
        levels = [int(x) for x in re.findall(r'\\(\s*-?\d+,\s*-?\d+,\s*-?\d+,\s*(-?\d+)\s*\\)', lv_s)]
        mine = 0 if "ai0_" in path.name else 1
        out.append((rnd, (c0, c1)[mine], own, sum(levels) / max(1, len(levels))))
    return out


def parse_ops(path: Path):
    """返回 [(round, side, [op_type...])]，side ∈ {'p0','p1'}。"""
    out = []
    if not path.exists():
        return out
    pat = re.compile(r"\[round (\d+)\] (p[01])\([^)]*\)[^\n]*?-> \[(.*?)\]\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), re.findall(r"OperationType\.(\w+)", m.group(3))))
    return out


stats = {"A": collections.defaultdict(list), "B": collections.defaultdict(list)}
ops_cnt = {"A": collections.Counter(), "B": collections.Counter()}
games = {"A": 0, "B": 0}

for seed in SEEDS:
    a_slot = "ai0" if seed % 2 == 0 else "ai1"
    for side_key, slot in (("A", a_slot), ("B", "ai1" if a_slot == "ai0" else "ai0")):
        lg = REPO / f"match_results/{slot}_az_bridge_ai_seed{seed}.stderr.log"
        rows = parse_bridge(lg)
        if len(rows) < 20:
            continue
        games[side_key] += 1
        for rnd, coins, own, lvl in rows:
            for lo, hi in BUCKETS:
                if lo <= rnd < hi:
                    stats[side_key][(lo, hi)].append((coins, own, lvl))
    # op 统计（A 侧 = a_slot 对应 p0/p1）
    for rnd, side, ops in parse_ops(AB / f"seed{seed}.log"):
        is_a = (side == "p0") == (a_slot == "ai0")
        key = "A" if is_a else "B"
        for op in ops:
            ops_cnt[key][op] += 1

print(f"样本：A 侧 {games['A']} 局 / B 侧 {games['B']} 局\n")
print(f"{'回合段':<12}{'A金币':>8}{'B金币':>8}{'差':>7}   {'A塔数':>8}{'B塔数':>8}{'差':>7}   {'A等级':>6}{'B等级':>6}")
for lo, hi in BUCKETS:
    a = stats["A"][(lo, hi)]
    b = stats["B"][(lo, hi)]
    if not a or not b:
        continue
    ac = statistics.mean(x[0] for x in a)
    bc = statistics.mean(x[0] for x in b)
    at = statistics.mean(x[1] for x in a)
    bt = statistics.mean(x[1] for x in b)
    al = statistics.mean(x[2] for x in a)
    bl = statistics.mean(x[2] for x in b)
    print(f"r{lo}-{hi:<7}{ac:>8.1f}{bc:>8.1f}{ac-bc:>+7.1f}   {at:>7.2f}{bt:>7.2f}{at-bt:>+7.2f}   {al:>6.2f}{bl:>6.2f}")

print("\nop 构成（A=有升级 / B=无升级）：")
keys = ["BUILD_TOWER", "DOWNGRADE_TOWER", "UPGRADE_TOWER", "USE_LIGHTNING_STORM"]
print(f"{'':<20}{'A':>10}{'B':>10}")
for k in keys:
    na, nb = ops_cnt["A"][k], ops_cnt["B"][k]
    ga = na / max(1, games["A"])
    gb = nb / max(1, games["B"])
    print(f"{k:<20}{ga:>10.1f}{gb:>10.1f}   （每局次数）")
