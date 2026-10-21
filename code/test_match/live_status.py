"""实时看正在进行的对局：每局的回合数、双方血量、金币、塔数。

数据源：`match_results/ai?_az_bridge_ai_seed*.stderr.log`
  · 开头 `[azai] init: player=N` → 我们执哪一侧
  · 每回合一行 `[azai] round=R hp=[a, b] coins=[c, d] towers=T lv=[...] lv_opp=[...]`
    （hp/coins 是**绝对玩家索引**，所以要按 player 取我们自己那一侧）

用法：python code/test_match/live_status.py [最近多少分钟，默认 40]
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

MR = Path("match_results")
RECENT_MIN = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
SEED_MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 22
SINCE = time.time() - RECENT_MIN * 60

pat_init = re.compile(r"\[azai\] init: player=(\d+) seed=(\d+)")
pat_round = re.compile(r"\[azai\] round=(\d+) hp=\[(\d+), (\d+)\] coins=\[(\d+), (\d+)\] "
                       r"towers=(\d+) lv=\[(.*?)\] lv_opp=\[(.*?)\]")

rows = []
skipped = 0
for log in sorted(MR.glob("ai?_az_bridge_ai_seed*.stderr.log")):
    if log.stat().st_mtime < SINCE:
        skipped += 1
        continue
    text = log.read_text(encoding="utf-8", errors="replace")
    m_init = pat_init.search(text)
    m_round = None
    for m in pat_round.finditer(text):
        m_round = m
    if not (m_init and m_round):
        continue
    player = int(m_init.group(1))
    seed = int(m_init.group(2))
    if seed > SEED_MAX:          # 只保留当前这场（种子 7..22）
        continue
    rnd = int(m_round.group(1))
    hp = [int(m_round.group(2)), int(m_round.group(3))]
    coins = [int(m_round.group(4)), int(m_round.group(5))]
    slots = len([x for x in m_round.group(7).split("),") if x.strip()])
    slots_opp = len([x for x in m_round.group(8).split("),") if x.strip()])
    ours, theirs = player, 1 - player
    rows.append((seed, player, rnd, hp[ours], hp[theirs],
                 coins[ours], coins[theirs], slots, slots_opp))

rows.sort(key=lambda r: (r[0], r[1]))
print(f"（已跳过 {skipped} 个历史日志；只统计最近 {RECENT_MIN:.0f} 分钟内写过的）")
print(f"{'seed':>5} {'执':>3} {'回合':>5} {'我血':>5} {'敌血':>5} "
      f"{'我金':>5} {'敌金':>5} {'我塔':>4} {'敌塔':>4} {'血量差':>6}")
lead = 0
for seed, player, rnd, hp_o, hp_t, c_o, c_t, s_o, s_t in rows:
    lead += 1 if hp_o > hp_t else 0
    print(f"{seed:>5} {'P'+str(player):>3} {rnd:>5} {hp_o:>5} {hp_t:>5} "
          f"{c_o:>5} {c_t:>5} {s_o:>4} {s_t:>4} {hp_o-hp_t:>+6}")
if rows:
    print(f"\n共 {len(rows)} 局在跑，其中我方血量领先 {lead} 局")
