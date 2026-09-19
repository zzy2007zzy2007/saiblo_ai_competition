#!/usr/bin/env python3
"""
weight_analysis.py — 权重/网络收敛分析
数据源 F: weight_stats.jsonl
分析内容:
  - 各层参数均值/标准差趋势 → 网络是否收敛
  - 梯度范数趋势 → 梯度消失/爆炸检测
  - 策略网络 vs 价值网络对比
"""
import json
import os
import sys
from collections import defaultdict, Counter

WEIGHT_PATH = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/training/weight_stats.jsonl"

def load_weight_stats():
    records = []
    with open(WEIGHT_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def main():
    records = load_weight_stats()
    print(f"权重统计记录数: {len(records)}\n")

    episodes = [r['episode'] for r in records]

    print("=" * 80)
    print("一、总体统计 (全程均值)")
    print("=" * 80)

    all_weight_means = []
    all_weight_stds = []
    all_grad_means = []
    all_grad_norms = []
    policy_grad_norms = []
    value_grad_norms = []

    for r in records:
        for wname, wdata in r['weights'].items():
            all_weight_means.append(wdata['mean'])
            all_weight_stds.append(wdata['std'])
            all_grad_means.append(wdata['grad_mean'])
            all_grad_norms.append(wdata['grad_norm'])
            if 'value_head' in wname:
                value_grad_norms.append(wdata['grad_norm'])
            else:
                policy_grad_norms.append(wdata['grad_norm'])

    print(f"  权重均值范围:      {min(all_weight_means):.4f} ~ {max(all_weight_means):.4f}")
    print(f"  权重标准差范围:    {min(all_weight_stds):.4f} ~ {max(all_weight_stds):.4f}")
    print(f"  梯度均值范围:      {min(all_grad_means):.4f} ~ {max(all_grad_means):.4f}")
    print(f"  梯度范数范围:      {min(all_grad_norms):.4f} ~ {max(all_grad_norms):.4f}")
    print(f"  策略网络平均梯度:  {sum(policy_grad_norms)/len(policy_grad_norms):.4f} (n={len(policy_grad_norms)})")
    print(f"  价值网络平均梯度:  {sum(value_grad_norms)/len(value_grad_norms):.4f} (n={len(value_grad_norms)})")

    print("\n" + "=" * 80)
    print("二、每层梯度范数趋势 (按 checkpoint 保存时间)")
    print("=" * 80)

    print(f"{'Episode':>8s} | {'总梯度范数':>8s} | {'策略梯度':>8s} | {'价值梯度':>8s} | {'最大层梯度':>10s} | {'参数量':>6s}")
    print("-" * 60)

    for r in records:
        ep = r['episode']
        total_norm = 0.0
        policy_norm = 0.0
        value_norm = 0.0
        max_layer = 0.0
        param_count = len(r['weights'])

        for wname, wdata in r['weights'].items():
            gn = wdata['grad_norm']
            total_norm += gn ** 2
            max_layer = max(max_layer, gn)
            if 'value_head' in wname:
                value_norm += gn ** 2
            else:
                policy_norm += gn ** 2

        total_norm = total_norm ** 0.5
        policy_norm = policy_norm ** 0.5
        value_norm = value_norm ** 0.5
        print(f"  Ep {ep:>4d} | {total_norm:>7.2f} | {policy_norm:>7.2f} | {value_norm:>7.2f} | {max_layer:>9.4f} | {param_count:>5d}")

    print("\n" + "=" * 80)
    print("三、网络收敛性判断")
    print("=" * 80)

    if len(records) >= 2:
        first_w = records[0]['weights']
        last_w = records[-1]['weights']

        param_deltas = []
        for wname in first_w:
            if wname in last_w:
                delta = abs(last_w[wname]['mean'] - first_w[wname]['mean'])
                param_deltas.append(delta)

        avg_delta = sum(param_deltas) / len(param_deltas)
        max_delta = max(param_deltas)
        print(f"  首尾参数量均值变化: {avg_delta:.6f}")
        print(f"  首尾参数量最大变化: {max_delta:.6f}")

        first_grads = [w['grad_norm'] for w in first_w.values()]
        last_grads = [w['grad_norm'] for w in last_w.values()]
        avg_first_grad = sum(first_grads) / len(first_grads)
        avg_last_grad = sum(last_grads) / len(last_grads)
        print(f"  初始平均梯度范数:   {avg_first_grad:.4f}")
        print(f"  最终平均梯度范数:   {avg_last_grad:.4f}")
        print(f"  梯度衰减比:         {avg_last_grad/avg_first_grad:.2f}x" if avg_first_grad > 0 else "  梯度: N/A")

        print(f"\n  判断: ", end="")
        if avg_last_grad < avg_first_grad * 0.1:
            print("✅ 梯度显著衰减，网络正在收敛")
        elif avg_last_grad < avg_first_grad * 0.5:
            print("🟡 梯度部分衰减，网络仍在学习")
        else:
            print("❌ 梯度未衰减，网络可能未收敛或仍在剧烈变化")

        if avg_delta < 0.001:
            print(f"  ✅ 参数变化很小 (均值变化 {avg_delta:.6f})，网络趋于稳定")
        else:
            print(f"  🟡 参数仍在变化 (均值变化 {avg_delta:.6f})，网络可能未稳定")

    print("\n" + "=" * 80)
    print("四、Top 10 最大梯度层 (最终 checkpoint)")
    print("=" * 80)

    if records:
        last_weights = records[-1]['weights']
        sorted_layers = sorted(last_weights.items(), key=lambda x: x[1]['grad_norm'], reverse=True)
        print(f"{'层名':40s} | {'权重均值':>8s} | {'权重标准差':>8s} | {'梯度均值':>8s} | {'梯度范数':>8s}")
        print("-" * 85)
        for wname, wdata in sorted_layers[:10]:
            print(f"  {wname:38s} | {wdata['mean']:>7.4f} | {wdata['std']:>7.4f} | {wdata['grad_mean']:>7.4f} | {wdata['grad_norm']:>7.4f}")

    print("\n" + "=" * 80)
    print("五、结论")
    print("=" * 80)
    print(f"""
  训练 checkpoint 数: {len(records)}
  总参数量:          {len(records[0]['weights']) if records else 0}
  策略网络:          {sum(1 for w in (records[0]['weights'] if records else {}) if 'value_head' not in w)}
  价值网络:          {sum(1 for w in (records[0]['weights'] if records else {}) if 'value_head' in w)}
""")

if __name__ == '__main__':
    main()
