#!/usr/bin/env python3
"""
从 league_state/ 生成对手池统计表。

数据来源:
  - {checkpoint_dir}/league_state/opponent_pool.json   — 对手 ID、加入 episode、对战场次
  - {checkpoint_dir}/league_state/payoff.json            — TrueSkill 评分、双向对战记录
  - {checkpoint_dir}/league_state/eviction_history.jsonl — 对手淘汰记录
  - {log_dir}/training/opponent_selection.log            — 对手选择频率
  - {log_dir}/training/per_episode_stats.jsonl           — 每局胜负结果
  - {log_dir}/selfplay/selfplay_battles/selfplay_battle_stats.json — 平均奖励、平均回合

用法:
  format_opponent_table.py /path/to/output_dir/
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime


def load_json(path, name):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_jsonl(path):
    records = []
    if not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def parse_opponent_selection(log_path):
    records = []
    if not os.path.exists(log_path):
        return records
    with open(log_path) as f:
        for line in f:
            m = re.search(r'Episode: (\d+).*Opponent: (\S+).*Random prob: ([\d.]+)', line)
            if m:
                records.append({
                    'episode': int(m.group(1)),
                    'opponent': m.group(2),
                    'prob': float(m.group(3))
                })
    return records


def format_opponent_table(output_dir: str) -> None:
    if not os.path.isdir(output_dir):
        print(f"错误：输出目录不存在 — {output_dir}", file=sys.stderr)
        sys.exit(1)

    dirname = os.path.basename(output_dir.rstrip('/'))
    start_time = "unknown"
    try:
        dt = datetime.strptime(dirname, "%Y%m%d_%H%M%S")
        start_time = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass

    checkpoint_dir = os.path.join(output_dir, "checkpoint")
    league_dir = os.path.join(checkpoint_dir, "league_state")
    selfplay_dir = os.path.join(output_dir, "selfplay", "selfplay_battles")
    training_dir = os.path.join(output_dir, "training")

    pool = load_json(os.path.join(league_dir, "opponent_pool.json"), "opponent_pool")
    payoff = load_json(os.path.join(league_dir, "payoff.json"), "payoff")
    battle_stats = load_json(os.path.join(selfplay_dir, "selfplay_battle_stats.json"), "selfplay_battle_stats")
    evictions = load_jsonl(os.path.join(league_dir, "eviction_history.jsonl"))
    selections = parse_opponent_selection(os.path.join(training_dir, "opponent_selection.log"))
    ep_stats = load_jsonl(os.path.join(training_dir, "per_episode_stats.jsonl"))

    print("=" * 80)
    print("对手池统计")
    print("=" * 80)

    max_size = pool.get('max_size', '?')
    opp_ids = pool.get('opponent_ids', [])
    added_episodes = pool.get('added_episodes', {})
    games_played = pool.get('games_played', {})
    trueskill = payoff.get('trueskill_ratings', {})
    battle_records = payoff.get('battle_records', {})

    print(f"训练目录:     {dirname}")
    print(f"启动时间:     {start_time}")
    print(f"对手池大小:   {len(opp_ids)}/{max_size}")
    print()

    # =====================================================
    # 1) 对手池详情表
    # =====================================================
    if not opp_ids:
        print("(对手池为空)")
    else:
        def _get_agent_stats(opp_id):
            wins = draws = losses = games = 0
            key = f"current_agent-{opp_id}"
            if key in battle_records:
                rec = battle_records[key]
                wins = rec.get('wins', 0)
                draws = rec.get('draws', 0)
                losses = rec.get('losses', 0)
                games = rec.get('games', 0)
            wr = (wins + 0.5 * draws) / games * 100 if games > 0 else 0
            return wins, draws, losses, games, wr

        header = (
            f"{'Agent ID':<22} {'加入Ep':>7} {'对局数':>6} {'胜':>4} {'平':>3} {'负':>4} "
            f"{'胜率':>6} {'Mu':>7} {'Sigma':>7} {'AvgReward':>9} {'AvgRound':>8}"
        )
        print(header)
        print("-" * len(header))

        for opp_id in opp_ids:
            ep = added_episodes.get(opp_id, 0)
            gp = games_played.get(opp_id, 0)
            wins, draws, losses, games, wr = _get_agent_stats(opp_id)
            rating = trueskill.get(opp_id, {})
            mu = rating.get('mu', 25.0)
            sigma = rating.get('sigma', 8.33)
            bs = battle_stats.get(opp_id, {})
            avg_rew = bs.get('avg_reward')
            avg_rnd = bs.get('avg_rounds')

            rew_str = f"{avg_rew:8.1f}" if avg_rew is not None else "      N/A"
            rnd_str = f"{avg_rnd:7.1f}" if avg_rnd is not None else "    N/A"
            ep_disp = ep if ep > 0 else "-"

            print(
                f"{opp_id:<22} {ep_disp:>7} {gp:>6} {wins:>4} {draws:>3} {losses:>4} "
                f"{wr:>5.1f}% {mu:>7.1f} {sigma:>7.2f} {rew_str:>9} {rnd_str:>8}"
            )

    # Current Agent
    ca_rating = trueskill.get('current_agent', {})
    ca_mu = ca_rating.get('mu', 25.0)
    ca_sigma = ca_rating.get('sigma', 8.33)
    ca_games = ca_rating.get('games_played', 0)
    print()
    print(f"Current Agent  —  Mu: {ca_mu:.1f}  Sigma: {ca_sigma:.2f}  Games: {ca_games}")

    # =====================================================
    # 2) TrueSkill 差距分析
    # =====================================================
    opp_ratings = []
    for oid in opp_ids:
        r = trueskill.get(oid, {})
        opp_ratings.append((oid, r.get('mu', 25.0), r.get('sigma', 8.33), games_played.get(oid, 0)))

    if opp_ratings:
        print()
        print("-" * 80)
        print("TrueSkill 评分对比")
        print("-" * 80)
        avg_opp_mu = sum(r[1] for r in opp_ratings) / len(opp_ratings)
        avg_opp_sigma = sum(r[2] for r in opp_ratings) / len(opp_ratings)
        gap_mu = ca_mu - avg_opp_mu

        print(f"  PPO TrueSkill Mu:        {ca_mu:.2f}")
        print(f"  对手平均 TrueSkill Mu:   {avg_opp_mu:.2f}")
        print(f"  差距 (PPO - 对手):       {gap_mu:+.2f}")
        print(f"  PPO Sigma:               {ca_sigma:.2f}")
        print(f"  对手平均 Sigma:          {avg_opp_sigma:.2f}")

        sorted_by_mu = sorted(opp_ratings, key=lambda x: -x[1])
        print(f"\n  对手强度排序 (由强到弱):")
        for oid, mu, sigma, gp in sorted_by_mu:
            label = f"{oid} (games={gp})"
            print(f"    Mu={mu:>6.2f}  Sigma={sigma:.2f}  {label}")

        top_n = max(3, len(sorted_by_mu) // 2)
        top_opps = sorted_by_mu[:top_n]
        top_wins = 0
        top_games = 0
        for oid, _, _, _ in top_opps:
            rec = battle_records.get(f"current_agent-{oid}", {})
            top_wins += rec.get('wins', 0)
            top_games += rec.get('games', 0)
        top_wr = top_wins / top_games * 100 if top_games > 0 else 0
        print(f"\n  对 top-{top_n} 对手合计胜率: {top_wr:.1f}%")

    # =====================================================
    # 3) 对手加入时间线
    # =====================================================
    print()
    print("-" * 80)
    print("对手加入时间线")
    print("-" * 80)
    if opp_ids:
        timeline = [(ep, oid) for oid in opp_ids if (ep := added_episodes.get(oid, 0)) > 0]
        if timeline:
            timeline.sort()
            for ep, oid in timeline:
                gp = games_played.get(oid, 0)
                print(f"  Ep {ep:>4d}: {oid}  (当前对局数={gp})")
        else:
            print("  (对手均在训练开始时加入，无后续添加)")
    else:
        print("  (对手池为空)")

    # =====================================================
    # 4) 对手淘汰记录
    # =====================================================
    if evictions:
        print()
        print("-" * 80)
        print("对手淘汰记录 (eviction_history.jsonl)")
        print("-" * 80)
        for ev in evictions:
            ts = ev.get('timestamp', '?')
            ep = ev.get('episode', '?')
            victim = ev.get('victim_id', '?')[:20]
            mu = ev.get('victim_mu', '?')
            score = ev.get('victim_final_score', '?')
            games = ev.get('victim_games', '?')
            pool_before = ev.get('pool_size_before', '?')
            pool_after = ev.get('pool_size_after', '?')
            print(f"  Ep {ep:<6} victim={victim:<22} mu={mu:<6} score={score:<6} games={games:<4} pool={pool_before}→{pool_after}")

    # =====================================================
    # 5) 对手选择频率 (opponent_selection.log)
    # =====================================================
    if selections:
        print()
        print("-" * 80)
        print("对手选择频率 (opponent_selection.log)")
        print("-" * 80)
        print(f"  选择记录数: {len(selections)}")

        opp_freq = Counter(s['opponent'] for s in selections)
        print(f"\n  {'对手ID':<25s} {'选择次数':>8s} {'占比':>8s}")
        print("  " + "-" * 45)
        for opp, cnt in opp_freq.most_common():
            print(f"  {opp:<25s} {cnt:>8d} {cnt/len(selections)*100:>7.1f}%")

        avg_prob = sum(s['prob'] for s in selections) / len(selections)
        print(f"\n  平均选择概率: {avg_prob:.4f}")

        ep_bins = defaultdict(list)
        for s in selections:
            lo = (s['episode'] // 200) * 200
            ep_bins[lo].append(s['prob'])
        if ep_bins:
            print(f"\n  选择概率随时间变化 (每 200 轮):")
            print(f"  {'Ep范围':>10s} | {'平均概率':>10s} | {'记录数':>6s}")
            print("  " + "-" * 32)
            for lo in sorted(ep_bins.keys()):
                probs = ep_bins[lo]
                print(f"  Ep {lo:>3d}-{lo+200:<4d} | {sum(probs)/len(probs):>9.4f} | {len(probs):>5d}")

    # =====================================================
    # 6) 每个对手的胜负详情 (per_episode_stats.jsonl)
    # =====================================================
    if ep_stats:
        print()
        print("-" * 80)
        print("每个对手的胜负详情 (per_episode_stats.jsonl)")
        print("-" * 80)
        opp_results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'draws': 0, 'total': 0, 'total_reward': 0.0})
        for r in ep_stats:
            opp_id = r.get('opponent_id', '?')
            result = r.get('result', '?')
            opp_results[opp_id]['total'] += 1
            opp_results[opp_id]['total_reward'] += r.get('reward', 0)
            if result == 'win':
                opp_results[opp_id]['wins'] += 1
            elif result == 'loss':
                opp_results[opp_id]['losses'] += 1
            elif result in ('draw', 'tie'):
                opp_results[opp_id]['draws'] += 1

        print(f"  {'对手ID':<25s} {'胜':>5s}-{'负':>5s}-{'平':>4s} {'胜率':>7s} {'局均Reward':>10s}")
        print("  " + "-" * 62)
        for opp in sorted(opp_results.keys()):
            d = opp_results[opp]
            wr = d['wins'] / d['total'] * 100 if d['total'] > 0 else 0
            avg_r = d['total_reward'] / d['total'] if d['total'] > 0 else 0
            print(f"  {opp[:23]:<25s} {d['wins']:>5d}-{d['losses']:>5d}-{d['draws']:>4d} {wr:>6.1f}% {avg_r:>9.1f}")

    # =====================================================
    # 7) 摘要
    # =====================================================
    total_wins = sum(r.get('wins', 0) for r in battle_records.values())
    total_games = sum(r.get('games', 0) for r in battle_records.values())
    overall_wr = total_wins / total_games * 100 if total_games > 0 else 0

    print()
    print("=" * 80)
    print("摘要")
    print("=" * 80)
    print(f"  对手池对手数:    {len(opp_ids)}")
    print(f"  对手选择记录:    {len(selections)} 条" if selections else "  对手选择记录:    无")
    print(f"  总对局数:        {total_games}")
    print(f"  总胜局数:        {total_wins} ({overall_wr:.1f}%)")
    print(f"  当前 PPO Mu:     {ca_mu:.2f}" if ca_rating else "  当前 PPO Mu:     无数据")
    print(f"  对手平均 Mu:     {avg_opp_mu:.2f}" if opp_ratings else "  对手平均 Mu:     无数据")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Format opponent pool statistics table')
    parser.add_argument('output_dir', help='Path to training output directory')
    args = parser.parse_args()
    format_opponent_table(args.output_dir)


if __name__ == '__main__':
    main()
