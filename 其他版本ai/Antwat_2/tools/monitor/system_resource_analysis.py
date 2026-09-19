#!/usr/bin/env python3
"""
system_resource_analysis.py — 系统资源分析
数据源 G: system_metrics.json.log
分析内容:
  - GPU 利用率趋势
  - 显存使用趋势
  - CPU/RAM 利用率
  - 训练阶段 vs 对战阶段对比
"""
import json
import os

SYSTEM_LOG_PATH = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/system/system_metrics.json.log"

def load_system_metrics():
    records = []
    with open(SYSTEM_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

def main():
    records = load_system_metrics()
    print(f"系统指标采样数: {len(records)}\n")

    print("=" * 80)
    print("一、总体资源使用统计")
    print("=" * 80)

    cpu_vals = [r.get('cpu_percent', 0) for r in records]
    ram_vals = [r.get('ram_percent', 0) for r in records]
    ram_used = [r.get('ram_used_gb', 0) for r in records]
    gpu_vals = [r.get('gpu_util', 0) for r in records]
    gpu_mem = [r.get('gpu_mem_used_mb', 0) for r in records]
    gpu_temp = [r.get('gpu_temp', 0) for r in records]

    print(f"  CPU 利用率:  平均 {sum(cpu_vals)/len(cpu_vals):.1f}%  最大 {max(cpu_vals):.1f}%")
    print(f"  RAM 利用率:  平均 {sum(ram_vals)/len(ram_vals):.1f}%  最大 {max(ram_vals):.1f}%")
    print(f"  RAM 占用:    平均 {sum(ram_used)/len(ram_used):.1f}GB  最大 {max(ram_used):.1f}GB")
    print(f"  GPU 利用率:  平均 {sum(gpu_vals)/len(gpu_vals):.1f}%  最大 {max(gpu_vals):.1f}%")
    print(f"  GPU 显存:    平均 {sum(gpu_mem)/len(gpu_mem):.0f}MB  最大 {max(gpu_mem):.0f}MB")
    print(f"  GPU 温度:    平均 {sum(gpu_temp)/len(gpu_temp):.0f}°C  最大 {max(gpu_temp):.0f}°C")

    training_records = [r for r in records if r.get('phase') == 'training']
    battle_records = [r for r in records if r.get('phase') == 'battle']

    print(f"\n  训练阶段采样: {len(training_records)}")
    print(f"  对战阶段采样: {len(battle_records)}")

    print("\n" + "=" * 80)
    print("二、训练阶段 vs 对战阶段对比")
    print("=" * 80)

    print(f"\n{'指标':>20s} | {'训练阶段':>10s} | {'对战阶段':>10s} | {'差异':>10s}")
    print("-" * 56)

    metrics_info = [
        ("CPU 均值", 'cpu_percent', training_records, battle_records),
        ("GPU 均值", 'gpu_util', training_records, battle_records),
        ("GPU 显存均值", 'gpu_mem_used_mb', training_records, battle_records),
        ("RAM 均值", 'ram_percent', training_records, battle_records),
    ]

    for label, key, tr, br in metrics_info:
        train_avg = sum(r.get(key, 0) for r in tr) / len(tr) if tr else 0
        battle_avg = sum(r.get(key, 0) for r in br) / len(br) if br else 0
        unit = "MB" if 'mem' in key else "%"
        diff = train_avg - battle_avg
        print(f"  {label:>20s} | {train_avg:>9.1f}{unit} | {battle_avg:>9.1f}{unit} | {diff:>+9.1f}{unit}")

    print("\n" + "=" * 80)
    print("三、GPU 利用率随时间变化 (每 100 条采样)")
    print("=" * 80)

    print(f"{'采样段':>10s} | {'GPU%':>6s} | {'显存MB':>8s} | {'CPU%':>6s} | {'RAM%':>6s} | {'阶段':>8s}")
    print("-" * 56)

    for i in range(0, len(records), max(1, len(records) // 20)):
        chunk = records[i:i + max(1, len(records) // 20)]
        if not chunk:
            continue
        avg_gpu = sum(r.get('gpu_util', 0) for r in chunk) / len(chunk)
        avg_mem = sum(r.get('gpu_mem_used_mb', 0) for r in chunk) / len(chunk)
        avg_cpu = sum(r.get('cpu_percent', 0) for r in chunk) / len(chunk)
        avg_ram = sum(r.get('ram_percent', 0) for r in chunk) / len(chunk)
        main_phase = chunk[len(chunk)//2].get('phase', '?')
        print(f"  {i:>4d}-{i+len(chunk):<4d} | {avg_gpu:>5.1f}% | {avg_mem:>7.0f} | {avg_cpu:>5.1f}% | {avg_ram:>5.1f}% | {main_phase:>8s}")

    print("\n" + "=" * 80)
    print("四、GPU 利用率分布")
    print("=" * 80)

    gpu_bins = [(0, 10), (10, 30), (30, 50), (50, 70), (70, 90), (90, 101)]
    print(f"{'GPU范围':>12s} | {'采样数':>8s} | {'占比':>8s}")
    print("-" * 34)
    for lo, hi in gpu_bins:
        count = sum(1 for r in records if lo <= r.get('gpu_util', 0) < hi)
        print(f"  {lo:>3d}%-{hi-1:<3d}%   | {count:>8d} | {count/len(records)*100:>7.1f}%")

    idle_count = sum(1 for r in records if r.get('gpu_util', 0) < 5)
    print(f"\n  GPU 空闲 (利用率<5%): {idle_count}/{len(records)} ({idle_count/len(records)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("五、结论")
    print("=" * 80)

    avg_gpu_all = sum(gpu_vals) / len(gpu_vals)
    if avg_gpu_all > 70:
        gpu_status = "✅ GPU 利用率较高"
    elif avg_gpu_all > 30:
        gpu_status = "🟡 GPU 利用率中等，可能有 CPU 瓶颈"
    else:
        gpu_status = "❌ GPU 利用率低，存在严重瓶颈"

    print(f"""
  GPU 平均利用率: {avg_gpu_all:.1f}% — {gpu_status}
  训练 vs 对战 GPU: {sum(r.get('gpu_util',0) for r in training_records)/len(training_records) if training_records else 0:.1f}% vs {sum(r.get('gpu_util',0) for r in battle_records)/len(battle_records) if battle_records else 0:.1f}%
  显存峰值: {max(gpu_mem):.0f}MB
""")

if __name__ == '__main__':
    main()
