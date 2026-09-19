#!/usr/bin/env python3
"""
time_analysis.py — 时间分配分析
数据源 K: time_statistics.json
分析内容:
  - 训练总时间 vs 对战评估总时间
  - 训练速度 (episodes/hour)
  - 各阶段的时间分配比例
"""
import json
import os
import re

TIME_STATS_PATH = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/system/time_statistics.json"
TRAINING_LOG_PATH = "/root/autodl-tmp/AntWar/ppo_v1/outputs/20260526_012851/training/training.log"

def load_time_stats():
    if not os.path.exists(TIME_STATS_PATH):
        return None
    with open(TIME_STATS_PATH) as f:
        return json.load(f)

def extract_training_speed(log_path):
    speeds = []
    if not os.path.exists(log_path):
        return speeds
    with open(log_path) as f:
        for line in f:
            if 'training_speed' in line and 'episode' in line:
                match = re.search(r'"training_speed":\s*([\d.]+)', line)
                if match:
                    speeds.append(float(match.group(1)))
            elif 'training_status.json' in line:
                match = re.search(r'"training_speed":\s*([\d.]+)', line)
                if match:
                    speeds.append(float(match.group(1)))
    return speeds

def main():
    stats = load_time_stats()

    print("=" * 70)
    print("一、时间统计概览")
    print("=" * 70)

    if stats:
        print(f"  训练总回合数:       {stats.get('training_episodes_count', '?')}")
        print(f"  训练总时间:         {stats.get('training_time_total', 0):.2f}s ({stats.get('training_time_total', 0)/3600:.2f}h)")
        print(f"  对战评估次数:       {stats.get('battle_evaluations_count', '?')}")
        print(f"  对战评估总时间:     {stats.get('battle_time_total', 0):.2f}s ({stats.get('battle_time_total', 0)/3600:.2f}h)")
        print(f"  最后一次对战时:     {stats.get('last_battle_duration', 0):.2f}s")

        train_time = stats.get('training_time_total', 0)
        battle_time = stats.get('battle_time_total', 0)
        total_time = train_time + battle_time

        if total_time > 0:
            print(f"\n  训练时间占比:       {train_time/total_time*100:.1f}%")
            print(f"  对战评估时间占比:   {battle_time/total_time*100:.1f}%")

        episodes = stats.get('training_episodes_count', 0)
        if episodes > 0 and train_time > 0:
            speed = episodes / (train_time / 3600)
            print(f"\n  训练速度:           {speed:.1f} episodes/hour")
            print(f"  平均每回合耗时:     {train_time/episodes:.2f}s/episode")

    else:
        print("  (time_statistics.json 不存在)")

    print("\n" + "=" * 70)
    print("二、训练速度趋势 (从 training.log 提取)")
    print("=" * 70)

    speeds = extract_training_speed(TRAINING_LOG_PATH)
    if speeds:
        print(f"  训练速度采样数: {len(speeds)}")
        print(f"  平均训练速度:   {sum(speeds)/len(speeds):.1f} episodes/hour")
        print(f"  最慢:           {min(speeds):.1f} episodes/hour")
        print(f"  最快:           {max(speeds):.1f} episodes/hour")
        print(f"  速度变化:       {max(speeds)-min(speeds):.1f} episodes/hour (波动幅度)")
    else:
        print("  (无法从日志提取训练速度)")

    print("\n" + "=" * 70)
    print("三、时间效率评估")
    print("=" * 70)

    if stats:
        train_time = stats.get('training_time_total', 0)
        battle_time = stats.get('battle_time_total', 0)
        total_time = train_time + battle_time

        battle_pct = battle_time / total_time * 100 if total_time > 0 else 0

        print(f"\n  评估维度:")
        print(f"    训练效率:   {train_time:.1f}s / {stats.get('training_episodes_count', 0)} episodes")

        if battle_pct > 50:
            print(f"    对战评估占比: {battle_pct:.1f}% — ❌ 对战评估时间占比过高，可能过多时间花在评估上")
        elif battle_pct > 20:
            print(f"    对战评估占比: {battle_pct:.1f}% — 🟡 对战评估时间占比适中")
        else:
            print(f"    对战评估占比: {battle_pct:.1f}% — ✅ 对战评估时间占比合理")

        if stats.get('battle_evaluations_count', 0) > 0:
            avg_battle_time = battle_time / stats['battle_evaluations_count']
            print(f"    每次对战评估平均耗时: {avg_battle_time:.1f}s")
            if avg_battle_time > 600:
                print(f"      — ❌ 每次评估超过 10 分钟，需要优化")
            else:
                print(f"      — ✅ 评估时间可接受")

    print("\n" + "=" * 70)
    print("四、关键时间节点")
    print("=" * 70)

    if stats:
        print(f"""
  训练开始:     {stats.get('last_training_start_time', 'N/A')}
  训练结束:     {stats.get('last_training_end_time', 'N/A')}
  最后一次对战: {stats.get('last_battle_start_time', 'N/A')} → {stats.get('last_battle_end_time', 'N/A')}
  训练持续时间: {stats.get('training_time_total', 0)/3600:.2f} 小时
""")

if __name__ == '__main__':
    main()
