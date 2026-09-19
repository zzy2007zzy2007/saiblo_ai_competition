import json, sys, math
from collections import Counter

def analyze(filepath):
    with open(filepath) as f:
        data = json.load(f)

    ep = data['episode']
    opp = data['opponent_id']
    result = data['result']
    total_reward = data['total_reward']
    rounds = data['rounds']
    duration = data.get('duration', 'N/A')
    start_time = data['start_time']
    end_time = data.get('end_time', 'N/A')

    details = data['round_details']
    final_hp = data['final_hp']
    action_counts = data.get('action_counts', {})
    actions = data.get('actions', [])

    # Core metrics
    our_coins_list = [r['our_coins'] for r in details]
    enemy_coins_list = [r['enemy_coins'] for r in details]
    our_hp_list = [r['our_hp'] for r in details]
    enemy_hp_list = [r['enemy_hp'] for r in details]
    our_towers_count = [len(r['our_towers']) for r in details]
    enemy_towers_count = [len(r['enemy_towers']) for r in details]

    max_coins = max(our_coins_list)
    end_coins = our_coins_list[-1]
    max_towers = max(our_towers_count)
    enemy_max_coins = max(enemy_coins_list)
    enemy_end_coins = enemy_coins_list[-1]
    enemy_max_towers = max(enemy_towers_count)

    # Reward analysis from actions list
    reward_list = []
    for action_entry in actions:
        reward = action_entry.get('reward', 0)
        if isinstance(reward, (int, float)):
            reward_list.append(reward)

    reward_mean = sum(reward_list) / len(reward_list) if reward_list else 0
    reward_var = sum((r - reward_mean) ** 2 for r in reward_list) / len(reward_list) if reward_list else 0
    reward_std = math.sqrt(reward_var)
    pos_rewards = sum(1 for r in reward_list if r > 0)
    neg_rewards = sum(1 for r in reward_list if r < 0)
    zero_rewards = sum(1 for r in reward_list if r == 0)

    # HP/Coins sampling every ~30 steps
    sample_interval = max(1, rounds // 30)
    hp_coin_traj = []
    for i in range(0, len(details), sample_interval):
        r = details[i]
        hp_coin_traj.append({
            'step': r['step'],
            'our_hp': r['our_hp'],
            'enemy_hp': r['enemy_hp'],
            'our_coins': r['our_coins'],
            'enemy_coins': r['enemy_coins'],
            'our_ants': r['our_ants'],
            'enemy_ants': r['enemy_ants'],
            'our_towers': len(r['our_towers']),
            'enemy_towers': len(r['enemy_towers']),
        })
    # Always include last step
    last = details[-1]
    hp_coin_traj.append({
        'step': last['step'],
        'our_hp': last['our_hp'],
        'enemy_hp': last['enemy_hp'],
        'our_coins': last['our_coins'],
        'enemy_coins': last['enemy_coins'],
        'our_ants': last['our_ants'],
        'enemy_ants': last['enemy_ants'],
        'our_towers': len(last['our_towers']),
        'enemy_towers': len(last['enemy_towers']),
    })

    print(f"=== Ep{ep} 分析报告 ===")
    print(f"基本信息:")
    print(f"  对手: {opp}")
    print(f"  结果: {result}")
    print(f"  回合数: {rounds}")
    print(f"  总奖励: {total_reward:.2f}")
    print(f"  用时: {duration}")
    print(f"  开始: {start_time}, 结束: {end_time}")
    print()
    print(f"核心指标:")
    print(f"  我方最大金币: {max_coins}, 终局金币: {end_coins}")
    print(f"  敌方最大金币: {enemy_max_coins}, 终局金币: {enemy_end_coins}")
    print(f"  我方最大塔数: {max_towers}")
    print(f"  敌方最大塔数: {enemy_max_towers}")
    print(f"  终局我方HP: {final_hp.get('our', 'N/A')}, 敌方HP: {final_hp.get('enemy', 'N/A')}")
    for key in ['first_move_wins', 'first_move_losses', 'second_move_wins', 'second_move_losses']:
        if key in data:
            print(f"  {key}: {data[key]}")
    print()
    print(f"动作分布 (action_counts):")
    total_actions = sum(action_counts.values()) if action_counts else 0
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action}: {count} ({pct:.1f}%)")
    if total_actions > 0:
        print(f"  总计: {total_actions}")
    print()
    print(f"奖励详情:")
    print(f"  均值: {reward_mean:.4f}")
    print(f"  标准差: {reward_std:.4f}")
    print(f"  正奖励: {pos_rewards}, 负奖励: {neg_rewards}, 零奖励: {zero_rewards}")
    print(f"  总数: {len(reward_list)}")
    print()
    print(f"HP/金币轨迹采样 (每~{sample_interval}步):")
    print(f"  {'Step':>5} | {'OurHP':>5} | {'EnemyHP':>5} | {'OurGold':>7} | {'EnemyGold':>7} | {'OurAnts':>7} | {'EnemyAnts':>7} | {'OurTowers':>9} | {'EnemyTowers':>9}")
    print(f"  {'-'*5} | {'-'*5} | {'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*9} | {'-'*9}")
    for t in hp_coin_traj:
        print(f"  {t['step']:>5} | {t['our_hp']:>5} | {t['enemy_hp']:>5} | {t['our_coins']:>7} | {t['enemy_coins']:>7} | {t['our_ants']:>7} | {t['enemy_ants']:>7} | {t['our_towers']:>9} | {t['enemy_towers']:>9}")

    print()
    # Win/loss status by position
    print(f"先手/后手胜负: first_move_wins={data.get('first_move_wins', 0)}, first_move_losses={data.get('first_move_losses', 0)}, second_move_wins={data.get('second_move_wins', 0)}, second_move_losses={data.get('second_move_losses', 0)}")
    print()

    return {
        'ep': ep,
        'opp': opp,
        'result': result,
        'rounds': rounds,
        'total_reward': total_reward,
        'max_coins': max_coins,
        'end_coins': end_coins,
        'max_towers': max_towers,
        'action_counts': action_counts,
        'reward_mean': reward_mean,
        'reward_std': reward_std,
        'pos_rewards': pos_rewards,
        'neg_rewards': neg_rewards,
        'zero_rewards': zero_rewards,
        'hp_coin_traj': hp_coin_traj,
    }

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_episode.py <json_file> [json_file2 ...]")
        sys.exit(1)
    for fp in sys.argv[1:]:
        analyze(fp)
        print("\n" + "=" * 80 + "\n")
