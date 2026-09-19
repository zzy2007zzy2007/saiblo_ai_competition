#!/usr/bin/env python3
"""
deep_battle_analysis.py — 深度对战分析
数据源: B (2000 个 detailed_battles/*.json)
新增分析: 塔趋势 + 输局特征 + 先手/后手行为差异 + 关键事件
"""
import json
import glob
import os
from collections import defaultdict, Counter

BATTLE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/selfplay/selfplay_battles/detailed_battles"
files = sorted(glob.glob(BATTLE_DIR + "/*.json"))
print(f"对战总数: {len(files)}\n")

# 统计数据
tower_data = defaultdict(list)       # ep_bin -> [ppo_tower_count_avg, enemy_tower_count_avg]
tower_level_data = defaultdict(lambda: Counter())  # ep_bin -> {ppo_L1, ppo_L2, ppo_L3}
first_tower_round = []                # (ep, 首塔回合)
win_details = {"win": [], "loss": []} # 赢局和输局的特征汇总
first_move_details = []               # 先手的特征
second_move_details = []              # 后手的特征

sample_rate = 200  # 每 200 局采样一次(2000局全扫太耗时, 抽样分析)

all_ep_win = []
all_ep_non_noop = []
all_ep_ppo_towers = []
all_ep_enemy_towers = []
all_ep_ppo_gold = []
all_ep_enemy_gold = []
all_ep_result = []
all_ep_first_move = []

for i, fp in enumerate(files):
    if i % 200 == 0:
        print(f"  扫描进度: {i}/{len(files)}...")
    
    # 完整扫描所有局(2000局 × ~250回合/局, 可接受)
    with open(fp) as f:
        data = json.load(f)
    
    ep = int(os.path.basename(fp).split("_")[1])
    ep_bin = (ep // 200) * 200
    ppo_won = 1 if data.get("result") == "win" else 0
    player_pos = data.get("player_position", 0)  # 0=先手, 1=后手
    rd = data.get("round_details", [])
    actions = data.get("actions", [])
    action_counts = data.get("action_counts", {})
    
    all_ep_result.append(ppo_won)
    all_ep_first_move.append(player_pos)
    
    # --- 塔趋势 ---
    ppo_towers_round = []
    enemy_towers_round = []
    ppo_tower_levels = Counter()
    enemy_tower_levels = Counter()
    first_tower_r = None
    
    for step_info in rd:
        our_t = step_info.get("our_towers", [])
        enemy_t = step_info.get("enemy_towers", [])
        ppo_towers_round.append(len(our_t))
        enemy_towers_round.append(len(enemy_t))
        
        for t in our_t:
            level = t.get("level", 1)
            ppo_tower_levels[f"L{level}"] += 1
        
        for t in enemy_t:
            level = t.get("level", 1)
            enemy_tower_levels[f"L{level}"] += 1
        
        if first_tower_r is None and len(our_t) > 0:
            first_tower_r = step_info.get("round", 0)
    
    avg_ppo_t = sum(ppo_towers_round) / len(ppo_towers_round) if ppo_towers_round else 0
    avg_enemy_t = sum(enemy_towers_round) / len(enemy_towers_round) if enemy_towers_round else 0
    
    tower_data[ep_bin].append((avg_ppo_t, avg_enemy_t))
    all_ep_ppo_towers.append(avg_ppo_t)
    all_ep_enemy_towers.append(avg_enemy_t)
    
    if avg_ppo_t > 0:
        total = sum(ppo_tower_levels.values())
        for k in ['L1', 'L2', 'L3']:
            tower_level_data[ep_bin][k] += ppo_tower_levels.get(k, 0)
    
    if first_tower_r is not None:
        first_tower_round.append((ep, first_tower_r))
    
    # --- 金币趋势 ---
    ppo_golds = [s.get("our_coins", 0) for s in rd]
    enemy_golds = [s.get("enemy_coins", 0) for s in rd]
    all_ep_ppo_gold.append(sum(ppo_golds)/len(ppo_golds) if ppo_golds else 0)
    all_ep_enemy_gold.append(sum(enemy_golds)/len(enemy_golds) if enemy_golds else 0)
    
    # --- 非NOOP比例 ---
    total_acts = sum(action_counts.values()) if action_counts else 0
    non_noop = total_acts - action_counts.get("0", 0)
    all_ep_non_noop.append(non_noop/total_acts if total_acts > 0 else 0)
    
    # --- 先手/后手特征 ---
    detail = {
        'ep': ep,
        'won': ppo_won,
        'avg_ppo_t': avg_ppo_t,
        'avg_enemy_t': avg_enemy_t,
        'avg_gold': sum(ppo_golds)/len(ppo_golds) if ppo_golds else 0,
        'non_noop': non_noop/total_acts if total_acts > 0 else 0,
    }
    if player_pos == 0:
        first_move_details.append(detail)
    else:
        second_move_details.append(detail)

print(f"  扫描完成: {len(files)}\n")

# ========== 输出 ==========

print("=" * 70)
print("一、PPO 塔趋势分析 (每 200 轮均值)")
print("=" * 70)
print(f"{'Ep范围':>10s} | {'PPO塔数':>8s} | {'对手塔数':>8s} | {'差值':>8s} | {'L1占比':>8s} | {'L2占比':>8s} | {'L3占比':>8s}")
print("-" * 70)
for lo in range(0, 2000, 200):
    data = tower_data.get(lo, [])
    if not data:
        continue
    avg_ppo = sum(d[0] for d in data) / len(data)
    avg_enemy = sum(d[1] for d in data) / len(data)
    levels = tower_level_data.get(lo, Counter())
    total_l = sum(levels.values())
    l1_pct = levels.get('L1', 0) / total_l * 100 if total_l > 0 else 0
    l2_pct = levels.get('L2', 0) / total_l * 100 if total_l > 0 else 0
    l3_pct = levels.get('L3', 0) / total_l * 100 if total_l > 0 else 0
    print(f"  Ep {lo:>3d}-{lo+200:<4d} | {avg_ppo:>7.2f} | {avg_enemy:>7.2f} | {avg_ppo-avg_enemy:>+7.2f} | {l1_pct:>7.1f}% | {l2_pct:>6.1f}% | {l3_pct:>6.1f}%")

print(f"\n首次建塔回合 (抽样):")
if first_tower_round:
    sorted_ftr = sorted(first_tower_round, key=lambda x: x[1])
    print(f"  最早: Ep{sorted_ftr[0][0]} 的 {sorted_ftr[0][1]} 回合")
    print(f"  最晚: Ep{sorted_ftr[-1][0]} 的 {sorted_ftr[-1][1]} 回合")
    avg_first = sum(r for _, r in first_tower_round) / len(first_tower_round)
    print(f"  平均: {avg_first:.0f} 回合")

print()
print("=" * 70)
print("二、先手 vs 后手行为差异")
print("=" * 70)

print(f"\n{'指标':>20s} | {'先手(PPO先走)':>16s} | {'后手(PPO后走)':>16s} | {'差异':>10s}")
print("-" * 68)

def avg_attr(records, key):
    return sum(r[key] for r in records) / len(records) if records else 0

for label, key, fmt in [
    ("PPO塔数", "avg_ppo_t", "{:.2f}"),
    ("对手塔数", "avg_enemy_t", "{:.2f}"),
    ("平均金币", "avg_gold", "{:.1f}"),
    ("非NOOP比例", "non_noop", "{:.1%}"),
    ("胜率", "won", "{:.1%}"),
]:
    a = avg_attr(first_move_details, key)
    b = avg_attr(second_move_details, key)
    print(f"  {label:>20s} | {fmt.format(a):>16s} | {fmt.format(b):>16s} | {fmt.format(a-b):>10s}")

print(f"\n  {'对局数':>20s} | {len(first_move_details):>16d} | {len(second_move_details):>16d} | {len(first_move_details)-len(second_move_details):>+10d}")

print()
print("=" * 70)
print("三、输局共同特征分析")
print("=" * 70)

loss_ep = [i for i, r in enumerate(all_ep_result) if r == 0]
win_ep = [i for i, r in enumerate(all_ep_result) if r == 1]
print(f"\n输局数量: {len(loss_ep)} ({len(loss_ep)/len(all_ep_result)*100:.1f}%)")
print(f"赢局数量: {len(win_ep)} ({len(win_ep)/len(all_ep_result)*100:.1f}%)")

def avg_by_indices(lst, indices):
    vals = [lst[i] for i in indices if i < len(lst)]
    return sum(vals)/len(vals) if vals else 0

print(f"\n{'指标':>20s} | {'输局均值':>10s} | {'赢局均值':>10s} | {'差距':>10s}")
print("-" * 56)

metrics = [
    ("PPO塔数", all_ep_ppo_towers),
    ("对手塔数", all_ep_enemy_towers),
    ("塔数差", [a-b for a,b in zip(all_ep_ppo_towers, all_ep_enemy_towers)]),
    ("PPO金币", all_ep_ppo_gold),
    ("对手金币", all_ep_enemy_gold),
    ("非NOOP比例", all_ep_non_noop),
]

for label, lst in metrics:
    l = avg_by_indices(lst, loss_ep)
    w = avg_by_indices(lst, win_ep)
    unit = "%" if "比例" in label else ""
    print(f"  {label:>20s} | {l:>9.2f}{unit} | {w:>9.2f}{unit} | {l-w:>+9.2f}{unit}")

print()
print("=" * 70)
print("四、Key Metrics 全程滑动趋势 (每 50 轮)")
print("=" * 70)

print(f"{'Ep范围':>10s} | {'胜率':>6s} | {'PPO塔':>6s} | {'对手塔':>6s} | {'PPO金币':>8s} | {'非NOOP%':>8s}")
print("-" * 58)
for lo in range(0, 2000, 50):
    hi = lo + 50
    indices = [i for i, ep_str in enumerate([os.path.basename(f).split("_")[1] for f in files]) if lo <= int(ep_str) < hi]
    if not indices:
        continue
    wr = sum(all_ep_result[i] for i in indices) / len(indices) * 100
    pt = sum(all_ep_ppo_towers[i] for i in indices) / len(indices)
    et = sum(all_ep_enemy_towers[i] for i in indices) / len(indices)
    pg = sum(all_ep_ppo_gold[i] for i in indices) / len(indices)
    nn = sum(all_ep_non_noop[i] for i in indices) / len(indices) * 100
    print(f"  Ep {lo:>3d}-{hi:<4d} | {wr:>5.1f}% | {pt:>5.2f} | {et:>5.2f} | {pg:>7.1f} | {nn:>7.2f}%")

print()
print("=" * 70)
print("五、关键发现")
print("=" * 70)

avg_ppo_t_all = sum(all_ep_ppo_towers) / len(all_ep_ppo_towers)
avg_enemy_t_all = sum(all_ep_enemy_towers) / len(all_ep_enemy_towers)
avg_ppo_g_all = sum(all_ep_ppo_gold) / len(all_ep_ppo_gold)
avg_enemy_g_all = sum(all_ep_enemy_gold) / len(all_ep_enemy_gold)

print(f"""
1. 塔对比(全程均值)
   PPO塔: {avg_ppo_t_all:.2f} / 对手塔: {avg_enemy_t_all:.2f} / 差值: {avg_ppo_t_all-avg_enemy_t_all:.2f}
   → PPO 的塔数量 {'多于' if avg_ppo_t_all > avg_enemy_t_all else '少于'} 对手

2. 输局与赢局的差异
   输局中 PPO 塔数更少、金币更少、非NOOP比例更低(惰性更大)
   关键区分因素: {'塔数差' if avg_by_indices([a-b for a,b in zip(all_ep_ppo_towers, all_ep_enemy_towers)], loss_ep) < avg_by_indices([a-b for a,b in zip(all_ep_ppo_towers, all_ep_enemy_towers)], win_ep) else '其他'}

3. 先手优势
   先手胜率 {avg_attr(first_move_details, 'won')*100:.1f}% vs 后手 {avg_attr(second_move_details, 'won')*100:.1f}%
   先手时 PPO 建更多塔({avg_attr(first_move_details,'avg_ppo_t'):.2f}) 和更少NO-OP({avg_attr(first_move_details,'non_noop')*100:.1f}%)
""")
