#!/usr/bin/env python3
"""
opponent_pool_analysis.py — 对手池动态分析
数据源 D (opponent_pool.json + payoff.json) + M (opponent_selection.log)
分析内容:
  - 对手的 TrueSkill 评分分布
  - 对手池扩充时间点
  - 对手选择频率
  - PPO 的可利用性趋势
"""
import json
import os
import re
from collections import defaultdict, Counter

LEAGUE_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/checkpoint/league_state"
LOG_DIR = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/training"

def load_opponent_pool():
    pool_path = os.path.join(LEAGUE_DIR, "opponent_pool.json")
    with open(pool_path) as f:
        return json.load(f)

def load_payoff():
    payoff_path = os.path.join(LEAGUE_DIR, "payoff.json")
    with open(payoff_path) as f:
        return json.load(f)

def load_opponent_selection():
    log_path = os.path.join(LOG_DIR, "opponent_selection.log")
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path) as f:
        for line in f:
            match = re.search(r'Episode: (\d+).*Opponent: (\S+).*Random prob: ([\d.]+)', line)
            if match:
                records.append({
                    'episode': int(match.group(1)),
                    'opponent': match.group(2),
                    'prob': float(match.group(3))
                })
    return records

def load_per_episode_stats():
    stat_path = os.path.join(LOG_DIR, "per_episode_stats.jsonl")
    records = {}
    if not os.path.exists(stat_path):
        return records
    with open(stat_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    records[r['episode']] = r
                except (json.JSONDecodeError, KeyError):
                    continue
    return records

def main():
    pool = load_opponent_pool()
    payoff = load_payoff()
    selections = load_opponent_selection()
    ep_stats = load_per_episode_stats()

    print("=" * 70)
    print("一、对手池概况")
    print("=" * 70)

    if isinstance(pool, dict):
        opponents = pool.get('opponents', pool)
        print(f"对手数量: {len(opponents)}")
        if isinstance(opponents, dict):
            for oid, oinfo in opponents.items():
                if isinstance(oinfo, dict):
                    ep = oinfo.get('episode_added', oinfo.get('episode', '?'))
                    print(f"  {oid}: 添加于 Ep {ep}")
                else:
                    print(f"  {oid}: {oinfo}")
    elif isinstance(pool, list):
        print(f"对手列表: {len(pool)} 个")
        for item in pool:
            print(f"  {item}")

    print("\n" + "=" * 70)
    print("二、对手 TrueSkill 评分 (payoff.json)")
    print("=" * 70)

    if isinstance(payoff, dict):
        players = payoff.get('players', payoff)
        print(f"\n{'对手ID':>20s} | {'Mu评分':>8s} | {'对局数':>6s} | {'胜率':>8s}")
        print("-" * 50)

        ratings = []
        if isinstance(players, dict):
            current_agent_rating = None
            for pid, pinfo in players.items():
                if isinstance(pinfo, dict):
                    mu = pinfo.get('mu', pinfo.get('rating', 0))
                    games = pinfo.get('games_played', pinfo.get('total_games', 0))
                    wins = pinfo.get('wins', 0)
                    wr = wins / games * 100 if games > 0 else 0
                    if pid == 'current_agent':
                        current_agent_rating = mu
                        print(f"  {'current_agent (PPO)':>20s} | {mu:>7.2f} | {games:>5d} | {wr:>7.1f}%")
                    else:
                        ratings.append((pid, mu, games, wr))

            for pid, mu, games, wr in sorted(ratings, key=lambda x: -x[1]):
                opp_name = pid[:18] if len(pid) > 18 else pid
                print(f"  {opp_name:>20s} | {mu:>7.2f} | {games:>5d} | {wr:>7.1f}%")

            if current_agent_rating is not None and ratings:
                avg_opp_mu = sum(r[1] for r in ratings) / len(ratings)
                print(f"\n  PPO TrueSkill: {current_agent_rating:.2f}")
                print(f"  对手平均 TrueSkill: {avg_opp_mu:.2f}")
                print(f"  差距: {current_agent_rating - avg_opp_mu:+.2f}")

    print("\n" + "=" * 70)
    print("三、对手选择记录分析 (opponent_selection.log)")
    print("=" * 70)

    if selections:
        print(f"对手选择记录数: {len(selections)}")

        opp_freq = Counter(s['opponent'] for s in selections)
        print(f"\n  对手选择频率:")
        for opp, cnt in opp_freq.most_common():
            print(f"    {opp:25s}: {cnt:>5d} 次 ({cnt/len(selections)*100:.1f}%)")

        avg_prob = sum(s['prob'] for s in selections) / len(selections)
        print(f"\n  平均选择概率: {avg_prob:.4f}")

        ep_bins = defaultdict(list)
        for s in selections:
            lo = (s['episode'] // 200) * 200
            ep_bins[lo].append(s['prob'])

        print(f"\n  选择概率随时间变化 (每 200 轮):")
        print(f"  {'Ep范围':>10s} | {'平均概率':>10s} | {'记录数':>6s}")
        print("-" * 32)
        for lo in sorted(ep_bins.keys()):
            probs = ep_bins[lo]
            print(f"  Ep {lo:>3d}-{lo+200:<4d} | {sum(probs)/len(probs):>9.4f} | {len(probs):>5d}")
    else:
        print("  (opponent_selection.log 不存在或格式不匹配)")

    print("\n" + "=" * 70)
    print("四、对手与 PPO 胜率的关系")
    print("=" * 70)

    if selections and ep_stats:
        opp_selection_by_ep = {}
        for s in selections:
            opp_selection_by_ep[s['episode']] = s['opponent']

        opp_win_rates = defaultdict(lambda: {'wins': 0, 'total': 0})

        for ep, stats in ep_stats.items():
            opp_id = stats.get('opponent_id', '?')
            result = stats.get('result', '?')
            if result == 'win':
                opp_win_rates[opp_id]['wins'] += 1
            if result in ('win', 'loss', 'draw'):
                opp_win_rates[opp_id]['total'] += 1

        print(f"\n{'对手ID':>25s} | {'胜/负':>10s} | {'胜率':>8s}")
        print("-" * 50)
        for opp in sorted(opp_win_rates.keys()):
            d = opp_win_rates[opp]
            wr = d['wins'] / d['total'] * 100 if d['total'] > 0 else 0
            losses = d['total'] - d['wins']
            print(f"  {opp[:23]:>25s} | {d['wins']:>3d}/{losses:<3d} | {wr:>7.1f}%")

    print("\n" + "=" * 70)
    print("五、结论")
    print("=" * 70)

    print(f"""
  对手池对手数: {len(pool) if isinstance(pool, (dict, list)) else '?'}
  对手选择记录: {len(selections)} 条
  对手强度梯度: {'存在分层' if len(ratings) > 1 else '数据不足'}
  PPO 利用性: {'待计算' if len(payoff) > 0 else '无数据'}
""")

if __name__ == '__main__':
    main()
