import json
import sys

metrics_file = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260528_144443/training/training_metrics_history.jsonl"
n_buckets = 4

with open(metrics_file) as f:
    entries = [json.loads(line) for line in f if line.strip()]

# =============================================
# 1. Reward 来源占比 (按 reward source 分类)
# =============================================
rw_keys = [k for k in entries[0] if k.startswith('rw_')]
half = len(entries) // 2

print("=" * 100)
print("Reward 来源占比分析")
print("=" * 100)

for label, bucket_entries in [("前期", entries[:half]), ("后期", entries[half:]), ("全部", entries)]:
    totals = {k: 0.0 for k in rw_keys}
    for e in bucket_entries:
        for k in rw_keys:
            totals[k] += e.get(k, 0.0)

    total = sum(totals.values()) or 1.0
    print(f"\n--- {label} ({len(bucket_entries)} 次 PPO update) ---")
    print(f"{'类别':<22} {'总额':>12} {'占比':>8} {'均额/update':>12}")
    print("-" * 60)
    for k in sorted(rw_keys):
        label_k = k[3:]
        print(f"{label_k:<22} {totals[k]:>12.1f} {totals[k]/total*100:>7.1f}% {totals[k]/max(len(bucket_entries),1):>12.1f}")
    print(f"{'--合计--':<22} {total:>12.1f} {'100.0%':>8}")

# =============================================
# 2. 各 reward source 随训练的趋势
# =============================================
print("\n" + "=" * 100)
print("Reward 来源随训练趋势 (每 PPO update 均额)")
print("=" * 100)

interval = max(1, len(entries) // n_buckets)
header = f"{'区间':>8}"
for k in sorted(rw_keys):
    header += f"  {k[3:]:>15}"
print(header)
print("-" * (8 + 18 * len(rw_keys)))

for bi in range(n_buckets):
    start = bi * interval
    end = min((bi + 1) * interval, len(entries))
    bucket = entries[start:end]
    avgs = {k: sum(e.get(k, 0.0) for e in bucket) / max(len(bucket), 1) for k in rw_keys}
    ep_range = f"E{bucket[0].get('episode',0)}-{bucket[-1].get('episode',0)}"
    line = f"{ep_range:>8}"
    for k in sorted(rw_keys):
        line += f"  {avgs[k]:>15.1f}"
    print(line)

# =============================================
# 3. Action 类型占比 (type_* 分布)
# =============================================
type_keys = [k for k in entries[0] if k.startswith('type_') and not k.startswith('type_valid_')]

print("\n" + "=" * 100)
print("Action 类型选择占比 (type_* 均值, 按区间)")
print("=" * 100)

header = f"{'区间':>8}"
for k in sorted(type_keys):
    header += f"  {k[5:]:>12}"
print(header)
print("-" * (8 + 14 * len(type_keys)))

for bi in range(n_buckets):
    start = bi * interval
    end = min((bi + 1) * interval, len(entries))
    bucket = entries[start:end]
    avgs = {k: sum(e.get(k, 0.0) for e in bucket) / max(len(bucket), 1) * 100 for k in type_keys}
    ep_range = f"E{bucket[0].get('episode',0)}-{bucket[-1].get('episode',0)}"
    line = f"{ep_range:>8}"
    for k in sorted(type_keys):
        line += f"  {avgs[k]:>11.1f}%"
    print(line)

# =============================================
# 4. 综合: 平均每步 reward 结构
# =============================================
print("\n" + "=" * 100)
print("平均每步 reward 结构 (最近一期 PPO update)")
print("=" * 100)

latest = entries[-1]
rw_total = sum(latest.get(k, 0.0) for k in rw_keys)
num_collected = latest.get('num_collected', 1)
avg_step_reward = rw_total / max(num_collected, 1)

print(f"  总步数 (num_collected): {num_collected}")
print(f"  总 reward: {rw_total:.1f}")
print(f"  平均每步 reward: {avg_step_reward:.4f}")
print()
print(f"{'类别':<22} {'总 reward':>12} {'占比':>8} {'每步均额':>12}")
print("-" * 60)
for k in sorted(rw_keys):
    val = latest.get(k, 0.0)
    per_step = val / max(num_collected, 1)
    print(f"{k[3:]:<22} {val:>12.1f} {val/rw_total*100:>7.1f}% {per_step:>12.4f}" if rw_total else
          f"{k[3:]:<22} {val:>12.1f} {'N/A':>8} {per_step:>12.4f}")
