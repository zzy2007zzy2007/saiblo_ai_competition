#!/usr/bin/env python3
"""
extract_training_trends.py — 训练指标全面分析
数据源: training_metrics_history.jsonl (~250 条记录)
包含: 训练趋势 + 奖励分解趋势 + 对手胜率曲线
"""
import json
import os
import sys
from collections import defaultdict, Counter

METRICS_PATH = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/training/training_metrics_history.jsonl"

def load_metrics():
    records = []
    with open(METRICS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    records = load_metrics()
    print(f"训练指标记录数: {len(records)}\n")

    # ---- 阶段 1: 每 200 轮分段统计 ----
    print("=" * 90)
    print("一、训练指标分段统计 (每 200 轮均值)")
    print("=" * 90)
    print(f"{'Ep范围':>10s} | {'Reward':>8s} | {'PolicyLoss':>10s} | {'Entropy':>8s} | {'LR':>10s} | {'Rounds':>7s} | {'GradNorm':>9s} | {'Ratio':>7s}")
    print("-" * 90)

    ep_bins = range(0, 2000, 200)
    for lo in ep_bins:
        hi = lo + 200
        bin_records = [r for r in records if lo <= r.get('episode', 0) < hi]
        if not bin_records:
            continue
        avg = lambda k: sum(r.get(k, 0) for r in bin_records) / len(bin_records)
        print(f"  Ep {lo:>3d}-{hi:<4d} | {avg('avg_reward'):>7.1f} | {avg('policy_loss'):>9.3f} | {avg('entropy'):>7.3f} | {avg('learning_rate'):>9.2e} | {avg('avg_rounds'):>6.0f} | {avg('gradient_norm'):>8.1f} | {avg('ratio_mean'):>6.2f}")

    # ---- 阶段 2: 奖励分解趋势 ----
    print("\n" + "=" * 90)
    print("二、奖励分解趋势 (每 200 轮均值)")
    print("=" * 90)
    reward_keys = ['rw_hp_attack_base', 'rw_hp_attack_tower', 'rw_coin_gain', 'rw_tower_survival', 'rw_die_penalty', 'rw_end_reward', 'rw_tech_bonus']
    header = f"{'Ep范围':>10s}"
    for k in reward_keys:
        header += f" | {k:>12s}"
    print(header)
    print("-" * (10 + len(reward_keys) * 16))

    for lo in ep_bins:
        hi = lo + 200
        bin_records = [r for r in records if lo <= r.get('episode', 0) < hi]
        if not bin_records:
            continue
        line = f"  Ep {lo:>3d}-{hi:<4d}"
        for k in reward_keys:
            v = sum(r.get(k, 0) for r in bin_records) / len(bin_records)
            line += f" | {v:>12.2f}"
        print(line)

    # ---- 阶段 3: 动作类型趋势 ----
    print("\n" + "=" * 90)
    print("三、动作类型占比趋势 (每 200 轮均值)")
    print("=" * 90)
    type_keys = sorted([k for k in records[0].keys() if k.startswith('type_')])
    if type_keys:
        header = f"{'Ep范围':>10s}"
        for k in type_keys:
            header += f" | {k:>12s}"
        print(header)
        print("-" * (10 + len(type_keys) * 16))
        for lo in ep_bins:
            hi = lo + 200
            bin_records = [r for r in records if lo <= r.get('episode', 0) < hi]
            if not bin_records:
                continue
            line = f"  Ep {lo:>3d}-{hi:<4d}"
            for k in type_keys:
                v = sum(r.get(k, 0) for r in bin_records) / len(bin_records) * 100
                line += f" | {v:>11.3f}%"
            print(line)
    else:
        print("  (训练指标中无 type_* 动作类型字段)")

    # ---- 阶段 4: 金币趋势 ----
    print("\n" + "=" * 90)
    print("四、金币趋势 (每 200 轮均值)")
    print("=" * 90)
    coin_keys = ['avg_our_coins', 'avg_enemy_coins']
    print(f"{'Ep范围':>10s} | {'avg_our_coins':>14s} | {'avg_enemy_coins':>16s} | {'差值':>8s}")
    print("-" * 60)
    for lo in ep_bins:
        hi = lo + 200
        bin_records = [r for r in records if lo <= r.get('episode', 0) < hi]
        if not bin_records:
            continue
        oc = sum(r.get('avg_our_coins', 0) for r in bin_records) / len(bin_records)
        ec = sum(r.get('avg_enemy_coins', 0) for r in bin_records) / len(bin_records)
        print(f"  Ep {lo:>3d}-{hi:<4d} | {oc:>14.2f} | {ec:>16.2f} | {oc-ec:>+7.2f}")

    # ---- 阶段 5: 对手胜率趋势 ----
    print("\n" + "=" * 90)
    print("五、对手胜率趋势 (每 200 轮)")
    print("=" * 90)
    opp_keys = sorted([k for k in records[0].keys() if 'opponent_win_rate' in k])
    if opp_keys:
        header = f"{'Ep范围':>10s}"
        for k in opp_keys:
            opp_name = k.replace('opponent_win_rate_', '')
            header += f" | {opp_name:>16s}"
        print(header)
        print("-" * (10 + len(opp_keys) * 20))
        for lo in ep_bins:
            hi = lo + 200
            bin_records = [r for r in records if lo <= r.get('episode', 0) < hi]
            if not bin_records:
                continue
            line = f"  Ep {lo:>3d}-{hi:<4d}"
            for k in opp_keys:
                v = sum(r.get(k, 0) for r in bin_records) / len(bin_records) * 100
                line += f" | {v:>15.1f}%"
            print(line)
    else:
        print("  (训练指标中无 opponent_win_rate_* 字段)")

    # ---- 阶段 6: 稳定性指标 ----
    print("\n" + "=" * 90)
    print("六、训练稳定性指标 (全程统计)")
    print("=" * 90)
    stable_keys = ['gradient_norm', 'ratio_mean', 'clip_fraction', 'valid_actions_mean', 'valid_actions_min', 'nan_skip_count']
    for k in stable_keys:
        vals = [r.get(k, 0) for r in records]
        print(f"  {k:20s}: 均值={sum(vals)/len(vals):.4f}, 最大={max(vals):.4f}, 最小={min(vals):.4f}")

    # ---- 阶段 7: 异常点检测 ----
    print("\n" + "=" * 90)
    print("七、异常点检测 (所有指标中 > 均值+3σ 的离群点)")
    print("=" * 90)
    check_keys = ['policy_loss', 'gradient_norm', 'ratio_mean']
    for k in check_keys:
        vals = [r.get(k, 0) for r in records]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean)**2 for v in vals) / len(vals))**0.5
        anomalies = [(r.get('episode'), r.get(k)) for r in records if abs(r.get(k, 0) - mean) > 3 * std]
        if anomalies:
            print(f"  {k:20s}: 均值={mean:.4f}, σ={std:.4f}, 阈值={mean+3*std:.4f}")
            print(f"          离群点: {', '.join(f'Ep{e}={v:.1f}' for e,v in anomalies[:10])}")
            if len(anomalies) > 10:
                print(f"          共 {len(anomalies)} 个离群点 (仅显示前10)")

    print("\n" + "=" * 90)
    print("八、对手胜率总体统计")
    print("=" * 90)
    if opp_keys:
        for k in opp_keys:
            vals = [r.get(k, 0) for r in records if r.get(k, 0) > 0]
            if vals:
                opp_name = k.replace('opponent_win_rate_', '')
                first_ep = next((r['episode'] for r in records if r.get(k, 0) > 0), '?')
                first_wr = next((r[k] for r in records if r.get(k, 0) > 0), 0)
                last_wr = [r[k] for r in records if r.get(k, 0) > 0][-1] if vals else 0
                avg_wr = sum(vals) / len(vals)
                print(f"  {opp_name:25s}: 首次={first_ep}Ep({first_wr*100:.1f}%), 最新={last_wr*100:.1f}%, 均值={avg_wr*100:.1f}%")

if __name__ == '__main__':
    main()
