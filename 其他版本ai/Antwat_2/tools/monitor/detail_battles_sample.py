#!/usr/bin/env python3
"""采样分析 detailed_battles 的金币峰值数据"""
import json, os

BASE = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260527_082648"
detail_dir = f"{BASE}/selfplay/selfplay_battles/detailed_battles/"
files = sorted(os.listdir(detail_dir))
print(f"文件总数: {len(files)}")

for idx in [0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]:
    with open(os.path.join(detail_dir, files[idx])) as f:
        data = json.load(f)
    rds = data.get("round_details", [])
    our_coins = [r.get("our_coins", 0) for r in rds]
    max_c = max(our_coins) if our_coins else 0
    p60 = 100*sum(1 for c in our_coins if c>=60)/len(our_coins) if our_coins else 0
    p200 = 100*sum(1 for c in our_coins if c>=200)/len(our_coins) if our_coins else 0
    ep = data.get("episode", "?")
    opp = (data.get("opponent_id", "?") or "?")[:20]
    result = data.get("result", "?")
    steps = len(rds)
    our_cum = data.get("our_cumulative_coins", 0)
    en_cum = data.get("enemy_cumulative_coins", 0)
    print(f"Ep={ep:>6d} opp={opp:>20s} result={result:>4s} steps={steps:>4d} max_balance={max_c:>4d} pct>=60={p60:>5.1f}% pct>=200={p200:>5.1f}% our_cum={our_cum:>6.0f} en_cum={en_cum:>6.0f}")
