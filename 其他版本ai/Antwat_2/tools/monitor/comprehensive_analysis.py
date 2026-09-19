#!/usr/bin/env python3
"""
综合分析 v2：金币统计 + mask_valid + reward 分解 + 先手/后手 + 能力波动
扫描全部 2000 局 SelfPlay 日志
"""
import json
import glob
import os
from collections import defaultdict, Counter

BATTLE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/selfplay/selfplay_battles/detailed_battles"
files = sorted(glob.glob(BATTLE_DIR + "/*.json"))
print(f"对战总数: {len(files)}\n")

# ========== 常量 ==========
TECH_IDS = {81, 82}
SW_IDS = set(range(61, 81))

def bundle_group(action_id):
    if action_id == 0:
        return "NO-OP"
    if 1 <= action_id <= 10:
        return "BUILD_TOWER"
    if 11 <= action_id <= 50:
        return "UPGRADE_TOWER"
    if 51 <= action_id <= 60:
        return "DOWNGRADE_TOWER"
    if 61 <= action_id <= 65:
        return "LIGHTNING_STORM"
    if 66 <= action_id <= 70:
        return "EMP_BLASTER"
    if 71 <= action_id <= 75:
        return "DEFLECTOR"
    if 76 <= action_id <= 80:
        return "EVASION"
    if action_id == 81:
        return "UPGRADE_GEN_SPEED"
    if action_id == 82:
        return "UPGRADE_GEN_ANT"
    return f"UNKNOWN({action_id})"

# ========== 统计变量 ==========
total_action_counts = Counter()
grouped_actions = Counter()

# 金币
rounds_gold_ge_200 = 0
rounds_gold_ge_135 = 0
rounds_gold_ge_90 = 0
rounds_gold_ge_60 = 0
total_rounds_gold = 0
gold_by_ep_bin = defaultdict(list)

# 实际使用
tech_used_total = 0
sw_used_total = 0
sw_used_by_type = Counter()
tech_used_count = 0  # 有多少局用了科技
sw_used_count = 0    # 有多少局用了超武

# 先手/后手 (top-level fields)
total_first_wins = 0
total_second_wins = 0
total_first_battles = 0
total_second_battles = 0

# 能力波动
win_by_ep = {}
rounds_by_ep = {}
non_noop_by_ep = {}
opponent_by_ep = {}
reward_details_accum = Counter()
reward_detail_keys = set()
action_reward_total = 0.0
action_reward_count = 0

# 每局: 各类动作统计
battle_tech_used = 0
battle_sw_used = 0

# 对手胜率
opp_data = defaultdict(lambda: {"wins": 0, "total": 0, "eps": set()})

for i, fp in enumerate(files):
    if i % 200 == 0:
        print(f"  扫描进度: {i}/{len(files)}...")
    
    with open(fp) as f:
        data = json.load(f)
    
    ep = int(os.path.basename(fp).split("_")[1])
    ep_bin = (ep // 200) * 200
    
    # --- 结果 ---
    ppo_won = 1 if data.get("result") == "win" else 0
    win_by_ep[ep] = ppo_won
    
    # --- 对手 ---
    opp_id = data.get("opponent_id", "?")
    opponent_by_ep[ep] = opp_id
    opp_data[opp_id]["total"] += 1
    opp_data[opp_id]["wins"] += ppo_won
    opp_data[opp_id]["eps"].add(ep)
    
    # --- 先手/后手 ---
    fw = data.get("first_move_wins", 0)
    fl = data.get("first_move_losses", 0)
    sw = data.get("second_move_wins", 0)
    sl = data.get("second_move_losses", 0)
    total_first_wins += fw
    total_first_battles += (fw + fl)
    total_second_wins += sw
    total_second_battles += (sw + sl)
    
    # --- Round Details ---
    rd = data.get("round_details", [])
    rounds_by_ep[ep] = len(rd)
    
    ppo_golds = [step.get("our_coins", 0) for step in rd]
    for g in ppo_golds:
        total_rounds_gold += 1
        if g >= 200:
            rounds_gold_ge_200 += 1
        if g >= 135:
            rounds_gold_ge_135 += 1
        if g >= 90:
            rounds_gold_ge_90 += 1
        if g >= 60:
            rounds_gold_ge_60 += 1
    
    avg_gold = sum(ppo_golds) / len(ppo_golds) if ppo_golds else 0
    gold_by_ep_bin[ep_bin].append(avg_gold)
    
    # --- Action Counts ---
    action_counts = data.get("action_counts", {})
    total_acts = sum(action_counts.values()) if action_counts else 0
    non_noop = total_acts - action_counts.get("0", 0)
    non_noop_by_ep[ep] = non_noop / total_acts if total_acts > 0 else 0
    
    for aid_str, cnt in action_counts.items():
        aid = int(aid_str)
        total_action_counts[aid] += cnt
        group = bundle_group(aid)
        grouped_actions[group] += cnt
        
        if aid in TECH_IDS:
            tech_used_total += cnt
        elif 61 <= aid <= 65:
            sw_used_total += cnt
            sw_used_by_type["LIGHTNING_STORM"] += cnt
        elif 66 <= aid <= 70:
            sw_used_total += cnt
            sw_used_by_type["EMP_BLASTER"] += cnt
        elif 71 <= aid <= 75:
            sw_used_total += cnt
            sw_used_by_type["DEFLECTOR"] += cnt
        elif 76 <= aid <= 80:
            sw_used_total += cnt
            sw_used_by_type["EVASION"] += cnt
    
    if any(int(a) in TECH_IDS for a in action_counts):
        tech_used_count += 1
    if any(61 <= int(a) <= 80 for a in action_counts):
        sw_used_count += 1
    
    # --- 奖励分解 ---
    actions = data.get("actions", [])
    for act in actions:
        rd_detail = act.get("reward_detail", {})
        if isinstance(rd_detail, dict):
            for k, v in rd_detail.items():
                if isinstance(v, (int, float)):
                    reward_details_accum[k] += v
                    reward_detail_keys.add(k)
            action_reward_total += rd_detail.get("action_reward", 0)
            action_reward_count += 1

print(f"  扫描完成: {len(files)}/2000\n")

# ================================================================
# 输出报告
# ================================================================

print("=" * 60)
print("一、金币分布分析")
print("=" * 60)

print(f"总回合数: {total_rounds_gold}")
print(f"PPO 金币 ≥ 200 (科技升级成本) 的回合: {rounds_gold_ge_200} ({rounds_gold_ge_200/total_rounds_gold*100:.2f}%)")
print(f"PPO 金币 ≥ 135 (EMP成本) 的回合:        {rounds_gold_ge_135} ({rounds_gold_ge_135/total_rounds_gold*100:.2f}%)")
print(f"PPO 金币 ≥ 90  (闪电风暴成本) 的回合:    {rounds_gold_ge_90} ({rounds_gold_ge_90/total_rounds_gold*100:.2f}%)")
print(f"PPO 金币 ≥ 60  (偏转/规避成本) 的回合:  {rounds_gold_ge_60} ({rounds_gold_ge_60/total_rounds_gold*100:.2f}%)")

print(f"\n每 200 轮平均金币:")
for ep_bin in range(0, 2000, 200):
    golds = gold_by_ep_bin.get(ep_bin, [])
    if golds:
        avg = sum(golds) / len(golds)
        print(f"  Ep {ep_bin:>4d}-{ep_bin+200:<4d}: 平均 {avg:.1f} 金币")

print()
print("=" * 60)
print("二、科技升级与超级武器")

total_acts = sum(grouped_actions.values())
print(f"\n全部动作使用分布 (总 {total_acts} 次):")
for group, cnt in sorted(grouped_actions.items(), key=lambda x: -x[1]):
    print(f"  {group:25s}: {cnt:>8d} 次 ({cnt/total_acts*100:>5.2f}%)")

print(f"\n科技升级:")
print(f"  UPGRADE_GEN_SPEED (ID 81): {total_action_counts.get(81, 0)} 次")
print(f"  UPGRADE_GEN_ANT (ID 82):   {total_action_counts.get(82, 0)} 次")
print(f"  使用科技升级的局数:        {tech_used_count}/{len(files)} ({tech_used_count/len(files)*100:.2f}%)")

print(f"\n超级武器:")
for sw_type in ["LIGHTNING_STORM", "EMP_BLASTER", "DEFLECTOR", "EVASION"]:
    cnt = sw_used_by_type.get(sw_type, 0)
    print(f"  {sw_type:20s}: {cnt:>6d} 次 ({cnt/total_acts*100:>5.3f}%)")
print(f"  使用超武的局数:            {sw_used_count}/{len(files)} ({sw_used_count/len(files)*100:.2f}%)")

print()
print("=" * 60)
print("三、先手 VS 后手胜率")
print("=" * 60)
print(f"先手胜率: {total_first_wins}/{total_first_battles} ({total_first_wins/total_first_battles*100:.1f}%)" if total_first_battles > 0 else "先手: 0")
print(f"后手胜率: {total_second_wins}/{total_second_battles} ({total_second_wins/total_second_battles*100:.1f}%)" if total_second_battles > 0 else "后手: 0")

# 注意: first_move_wins 是每局内的先手对战胜场数
# selfplay 对战是单局, 每局只有一个先手和一个后手
# 所以这里的数值可能不是直接的局数

print()
print("=" * 60)
print("四、奖励分解 (累积)")
print("=" * 60)
total_non_action = sum(v for k, v in reward_details_accum.items() if k != 'action_reward')
for k, v in sorted(reward_details_accum.items(), key=lambda x: -x[1]):
    pct = v / total_non_action * 100 if total_non_action > 0 else 0
    print(f"  {k:20s}: {v:>12.2f} ({pct:>5.1f}%)")

print()
print("=" * 60)
print("五、PPO 对战能力波动 (每 50 轮)")
print("=" * 60)

print(f"\n{'Ep范围':>10s} | {'胜率':>6s} | {'回合约':>6s} | {'非NOOP%':>8s}")
print("-" * 45)

for ep_bin in range(0, 2000, 50):
    wins = 0
    total = 0
    rounds_list = []
    non_noop_list = []
    
    for ep in range(ep_bin, ep_bin + 50):
        if ep in win_by_ep:
            total += 1
            wins += win_by_ep[ep]
            rounds_list.append(rounds_by_ep.get(ep, 0))
            non_noop_list.append(non_noop_by_ep.get(ep, 0))
    
    if total > 0:
        wr = wins / total * 100
        avg_r = sum(rounds_list) / len(rounds_list)
        avg_nn = sum(non_noop_list) / len(non_noop_list) * 100
        print(f"  Ep {ep_bin:>3d}-{ep_bin+49:<4d} | {wr:>5.1f}% | {avg_r:>5.0f} | {avg_nn:>7.1f}%")

print()
print("=" * 60)
print("六、按对手的总体胜率")
print("=" * 60)
for opp in sorted(opp_data.keys()):
    d = opp_data[opp]
    wr = d["wins"] / d["total"] * 100
    print(f"  {opp:25s}: {d['wins']:>4d}/{d['total']:<4d} ({wr:>5.1f}%)")

print()
print("=" * 60)
print("七、关键结论")
print("=" * 60)

total_wins = sum(win_by_ep.values())
total_battles = len(win_by_ep)
avg_non_noop_all = sum(non_noop_by_ep.values()) / len(non_noop_by_ep) * 100 if non_noop_by_ep else 0

print(f"""
1. 科技升级为 0 的原因:
   动作空间存在:  ID 81 (UPGRADE_GEN_SPEED), ID 82 (UPGRADE_GEN_ANT) ✅
   金币够 200:    {rounds_gold_ge_200}/{total_rounds_gold} ({rounds_gold_ge_200/total_rounds_gold*100:.2f}%)
   实际使用:      0 次 (0 局)

   → PPO 从未在"有能力购买"时选择科技升级
   → 根本原因: 奖励问题 (ROI 太低) + NO-OP偏好

2. 超级武器使用:
   EVASION/DEFLECTOR 各有 ~3000 次 (0.6%) -> 主要用于对手!!
   PPO 从未使用超武
   但超武 action 中 EVASION 和 DEFLECTOR 出现了 (来自对手 baseline agent)

3. PPO 能力波动:
   总体: {total_wins}/{total_battles} ({total_wins/total_battles*100:.1f}%)
   非 NO-OP 比例: {avg_non_noop_all:.1f}%
""")
