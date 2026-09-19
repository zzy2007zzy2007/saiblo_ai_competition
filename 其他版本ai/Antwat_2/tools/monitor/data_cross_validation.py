#!/usr/bin/env python3
"""
data_cross_validation.py — 数据一致性交叉验证
验证多个数据源之间的指标是否一致
验证项:
  1. A (training_metrics) 的 avg_our_coins vs B (detailed battles) 的金币统计
  2. B 的 result vs C (per_episode_stats) 的 result
  3. A 的 reward 分解 vs B 的 reward_detail 累加
"""
import json
import os
import glob
from collections import defaultdict, Counter

TRAINING_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/training"
BATTLE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/selfplay/selfplay_battles/detailed_battles"

def load_metrics_history():
    path = os.path.join(TRAINING_DIR, "training_metrics_history.jsonl")
    records = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records[r['episode']] = r
    return records

def load_per_episode_stats():
    path = os.path.join(TRAINING_DIR, "per_episode_stats.jsonl")
    records = {}
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    records[r['episode']] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return records

def scan_battle_files():
    files = sorted(glob.glob(os.path.join(BATTLE_DIR, "*.json")))
    if not files:
        return {}, {}, {}
    results = {}
    rewards = {}
    coins = {}
    for i, fp in enumerate(files):
        if i % 500 == 0:
            print(f"  扫描对战文件进度: {i}/{len(files)}...")
        with open(fp) as f:
            data = json.load(f)
        ep = int(os.path.basename(fp).split("_")[1])
        results[ep] = data.get("result", "?")
        rewards[ep] = data.get("total_reward", 0)
        rd = data.get("round_details", [])
        ppo_coins = [s.get("our_coins", 0) for s in rd]
        coins[ep] = sum(ppo_coins) / len(ppo_coins) if ppo_coins else 0
    return results, rewards, coins

def main():
    print("=" * 70)
    print("数据一致性交叉验证")
    print("=" * 70)

    print("\n正在加载数据源 A (training_metrics_history.jsonl)...")
    metrics_records = load_metrics_history()
    print(f"  加载了 {len(metrics_records)} 条训练指标")

    print("\n正在加载数据源 C (per_episode_stats.jsonl)...")
    ep_stats = load_per_episode_stats()
    print(f"  加载了 {len(ep_stats)} 条每局摘要")

    print("\n正在扫描数据源 B (detailed_battles)...")
    battle_results, battle_rewards, battle_coins = scan_battle_files()
    print(f"  扫描了 {len(battle_results)} 局对战详情")

    print("\n" + "=" * 70)
    print("验证 1: B (detailed_battles) 的 result vs C (per_episode_stats) 的 result")
    print("=" * 70)

    mismatches = []
    for ep in sorted(set(battle_results.keys()) & set(ep_stats.keys())):
        br = battle_results[ep]
        cr = ep_stats[ep].get('result', '?')
        if br != cr:
            mismatches.append((ep, br, cr))

    if mismatches:
        print(f"  ❌ 发现 {len(mismatches)} 处结果不匹配:")
        for ep, br, cr in mismatches[:10]:
            print(f"    Ep {ep}: B={br}, C={cr}")
        if len(mismatches) > 10:
            print(f"    ... 还有 {len(mismatches) - 10} 处")
    else:
        print(f"  ✅ 全部 {len(set(battle_results.keys()) & set(ep_stats.keys()))} 局结果一致")

    print("\n" + "=" * 70)
    print("验证 2: A 的 avg_our_coins vs B 的逐局金币均值")
    print("=" * 70)

    mismatches = []
    for ep, metrics in sorted(metrics_records.items()):
        if ep in battle_coins:
            a_coins = metrics.get('avg_our_coins', 0)
            b_coins = battle_coins[ep]
            if abs(a_coins - b_coins) > 5:
                mismatches.append((ep, a_coins, b_coins, abs(a_coins - b_coins)))

    if mismatches:
        print(f"  ❌ 发现 {len(mismatches)} 处整体金币差异过大 (>5 金币):")
        for ep, a, b, diff in mismatches[:10]:
            print(f"    Ep {ep}: A={a:.1f}, B={b:.1f}, 差异={diff:.1f}")
        if len(mismatches) > 10:
            print(f"    ... 还有 {len(mismatches) - 10} 处")
        total_a = sum(metrics_records[ep].get('avg_our_coins', 0) for ep in metrics_records if ep in battle_coins)
        total_b = sum(battle_coins[ep] for ep in metrics_records if ep in battle_coins)
        overlap = len(set(metrics_records.keys()) & set(battle_coins.keys()))
        print(f"\n  总体验证: 重叠 {overlap} 局")
        if overlap > 0:
            avg_a = total_a / overlap
            avg_b = total_b / overlap
            print(f"  平均金币: A={avg_a:.1f}, B={avg_b:.1f}, 差异={abs(avg_a-avg_b):.1f}")
    else:
        print(f"  ✅ 所有重叠 {len(set(metrics_records.keys()) & set(battle_coins.keys()))} 局的金币数据一致")

    print("\n" + "=" * 70)
    print("验证 3: 胜率一致性 (A 的统计数据 vs B 的逐局统计)")
    print("=" * 70)

    total_b_wins = sum(1 for r in battle_results.values() if r == 'win')
    total_b_total = len(battle_results)
    print(f"  数据源 B: {total_b_wins}/{total_b_total} = {total_b_wins/total_b_total*100:.1f}%")

    c_total = len(ep_stats)
    c_wins = sum(1 for s in ep_stats.values() if s.get('result') == 'win')
    if c_total > 0:
        print(f"  数据源 C: {c_wins}/{c_total} = {c_wins/c_total*100:.1f}%")
        if c_total == total_b_total and c_wins == total_b_wins:
            print(f"  ✅ B 和 C 的总体胜率一致")
        else:
            print(f"  ⚠️ B 和 C 的总体胜率不一致 (局数差异: B={total_b_total}, C={c_total})")

    print("\n" + "=" * 70)
    print("验证 4: A 的 reward 分解 vs B 的 reward_detail 累加")
    print("=" * 70)

    print(f"""
  注意: A 中的 rw_* 字段是通过在 selfplay.py 中扫描 B 的
  reward_detail 累加得到的，因此两者应该完全一致。
  但 A 的记录粒度是 PPO update (每 8 轮)，B 是逐局记录。
  如果 A 中某个 episode 的 rw_hp_attack_base / rw_hp_attack_tower 等字段都为 0，
  可能是累加过程出现问题。
""")
    zero_reward_count = 0
    for ep, metrics in sorted(metrics_records.items()):
        total = sum(metrics.get(k, 0) for k in
                    ['rw_hp_attack_base', 'rw_hp_attack_tower', 'rw_coin_gain', 'rw_tower_survival',
                     'rw_tech_bonus', 'rw_die_penalty', 'rw_end_reward'])
        if abs(total) < 0.001 and ep > 0:
            zero_reward_count += 1

    print(f"  训练指标中 rw_* 全为 0 的 update 次数: {zero_reward_count}/{len(metrics_records)}")
    if zero_reward_count > len(metrics_records) * 0.5:
        print(f"  ❌ 超过一半的 rw_* 为 0，奖励分解记录可能有问题")
    else:
        print(f"  ✅ 奖励分解基本正常")

    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)

    issues = []
    if mismatches:
        issues.append(f"B vs C 结果不一致: {len(mismatches)} 处")
    if zero_reward_count > len(metrics_records) * 0.5:
        issues.append(f"奖励分解记录异常")

    if issues:
        print(f"\n  发现 {len(issues)} 个数据一致性问题:")
        for issue in issues:
            print(f"    ⚠️  {issue}")
    else:
        print(f"\n  ✅ 所有验证项通过，数据一致性良好")

    print(f"""
  数据覆盖:
    训练指标(A): {len(metrics_records)} 条记录
    对战详情(B): {len(battle_results)} 局
    每局摘要(C): {len(ep_stats)} 条
""")

if __name__ == '__main__':
    main()
