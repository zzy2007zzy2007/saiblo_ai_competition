#!/usr/bin/env python3
"""第四次训练评估 - 提取疑点数据"""
import json

METRICS = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260527_082648/training/training_metrics_history.jsonl"

entries = []
with open(METRICS) as f:
    for line in f:
        entries.append(json.loads(line.strip()))

# ============ 动作分布 ============
print("=" * 70)
print("动作分布趋势")
print("=" * 70)
action_keys = sorted([k for k in entries[0] if k.startswith("type_") and "valid" not in k])
print(f"{'动作':<20s} {'整体均值':>8s} {'前1/3':>8s} {'后1/3':>8s} {'趋势':>6s}")
print("-" * 52)
n = len(entries)
for ak in action_keys:
    vals = [e.get(ak, 0) for e in entries]
    avg = sum(vals) / n
    f3 = sum(vals[:n//3]) / (n//3)
    l3 = sum(vals[-n//3:]) / (n//3)
    if l3 > f3 * 1.1:
        trend = "↑上升"
    elif l3 < f3 * 0.9:
        trend = "↓下降"
    else:
        trend = "→持平"
    name = ak.replace("type_", "")
    print(f"{name:<20s} {avg:>8.2f} {f3:>8.2f} {l3:>8.2f} {trend:>6s}")

# ============ 学习率 vs entropy ============
print("\n" + "=" * 70)
print("学习率 / Entropy / EntCoef 趋势")
print("=" * 70)
print(f"{'Ep':>6s} {'LR':>12s} {'Entropy':>8s} {'EntCoef':>8s}")
print("-" * 38)
for i in range(0, len(entries), 20):
    e = entries[i]
    ep = e.get("episode", 0)
    lr = f"{e.get('learning_rate', 0):.10f}"
    ent = e.get("entropy", 0)
    ec = e.get("entropy_coef", 0)
    print(f"{ep:>6d} {lr:>12s} {ent:>8.4f} {ec:>8.4f}")

# ============ 累计金币 ============
print("\n" + "=" * 70)
print("累计金币统计 (avg_our_cumulative_coins)")
print("=" * 70)
cum_vals = [e.get("avg_our_cumulative_coins", 0) for e in entries]
cum_enemy = [e.get("avg_enemy_cumulative_coins", 0) for e in entries]
if any(cum_vals):
    avg_our = sum(cum_vals) / len(cum_vals)
    avg_enemy = sum(cum_enemy) / len(cum_enemy)
    print(f"我方累计金币: mean={avg_our:.1f}  min={min(cum_vals):.1f}  max={max(cum_vals):.1f}")
    print(f"敌方累计金币: mean={avg_enemy:.1f}  min={min(cum_enemy):.1f}  max={max(cum_enemy):.1f}")
    # 趋势
    first_half = cum_vals[:len(cum_vals)//2]
    second_half = cum_vals[len(cum_vals)//2:]
    print(f"前半段均值: {sum(first_half)/len(first_half):.1f}")
    print(f"后半段均值: {sum(second_half)/len(second_half):.1f}")
    diff = [o-e for o,e in zip(cum_vals, cum_enemy)]
    net_win = sum(1 for d in diff if d > 0)
    net_loss = sum(1 for d in diff if d < 0)
    print(f"经济领先局: {net_win}/{len(diff)} ({100*net_win/len(diff):.1f}%)")
    print(f"经济落后局: {net_loss}/{len(diff)} ({100*net_loss/len(diff):.1f}%)")
else:
    print("(数据不可用)")

# ============ 梯度异常率 ============
print("\n" + "=" * 70)
print("梯度稳定性")
print("=" * 70)
grads = [e.get("gradient_norm", 0) for e in entries]
grad_v = [e.get("grad_norm_value", 0) for e in entries]
n_gt_200 = sum(1 for g in grads if g > 200)
n_gt_500 = sum(1 for g in grads if g > 500)
n_gt_1000 = sum(1 for g in grads if g > 1000)
nv_gt_200 = sum(1 for g in grad_v if g > 200)
print(f"总更新次数: {len(grads)}")
print(f"gradient_norm > 200: {n_gt_200}/{len(grads)} ({100*n_gt_200/len(grads):.1f}%)")
print(f"gradient_norm > 500: {n_gt_500}/{len(grads)} ({100*n_gt_500/len(grads):.1f}%)")
print(f"gradient_norm > 1000: {n_gt_1000}/{len(grads)} ({100*n_gt_1000/len(grads):.1f}%)")
print(f"grad_norm_value > 200: {nv_gt_200}/{len(grad_v)} ({100*nv_gt_200/len(grad_v):.1f}%)")
print(f"梯度均值: {sum(grads)/len(grads):.1f}  峰值: {max(grads):.1f}")

# ============ 奖励趋势 ============
print("\n" + "=" * 70)
print("奖励趋势 (每20 ep 采样)")
print("=" * 70)
rewards = [e.get("avg_reward", 0) for e in entries]
episodes = [e.get("episode", i+1) for i in range(len(entries))]
print(f"整体: mean={sum(rewards)/len(rewards):.1f}  std={__import__('statistics').stdev(rewards):.1f}")
r_first = rewards[:n//3]
r_last = rewards[-n//3:]
print(f"前1/3: mean={sum(r_first)/len(r_first):.1f}")
print(f"后1/3: mean={sum(r_last)/len(r_last):.1f}")
