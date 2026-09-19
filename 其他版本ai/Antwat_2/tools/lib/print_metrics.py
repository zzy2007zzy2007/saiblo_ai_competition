#!/usr/bin/env python3
"""Helper script to display training metrics and system metrics from JSON files."""

import json
import sys

def print_training_metrics(metrics_file):
    """Print training metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        latest = data.get('latest', data)

        print("=" * 70)
        print("[Training Metrics Summary]")
        print("=" * 70)
        print("  Episode:       " + str(latest.get('episode', 'N/A')))
        print("  Steps:         " + str(latest.get('steps', 0)))
        print("  Avg Reward:    " + str(round(latest.get('avg_reward', 0), 4)))
        print("  Avg Loss:      " + str(round(latest.get('avg_loss', 0), 6)))
        print("  Avg Rounds:    " + str(round(latest.get('avg_rounds', 0), 1)))
        print("  Policy Loss:   " + str(round(latest.get('policy_loss', 0), 6)))
        print("  Value Loss:    " + str(round(latest.get('value_loss', 0), 6)))
        print("  Entropy:       " + str(round(latest.get('entropy', 0), 6)))
        print("  Entropy Coef:  " + str(round(latest.get('entropy_coef', 0), 6)))
        print("  Learning Rate: " + str(round(latest.get('learning_rate', 0), 8)))

        if 'stats' in data:
            stats = data['stats']
            print("=" * 70)
            print("[Historical Stats]")
            print("=" * 70)
            print("  Reward Mean:  " + str(round(stats.get('reward_mean', 0), 4)))
            print("  Reward Std:   " + str(round(stats.get('reward_std', 0), 4)))
            print("  Reward Max:   " + str(round(stats.get('reward_max', 0), 4)))
            print("  Reward Min:   " + str(round(stats.get('reward_min', 0), 4)))

        history = data.get('history', [])
        if history:
            print()
            print("[Last 10 Episodes]")
            print("-" * 70)
            print("{:>6} {:>12} {:>14} {:>10}".format("Ep", "Reward", "Loss", "Steps"))
            print("-" * 70)
            for h in history[-10:]:
                print("{:>6} {:>12.4f} {:>14.6f} {:>10,}".format(
                    h.get('episode', 0), h.get('avg_reward', 0),
                    h.get('avg_loss', 0), h.get('steps', 0)
                ))
            print("-" * 70)

        print()
        print("=" * 70)

    except Exception as e:
        print("Error: " + str(e))
        sys.exit(1)


def print_system_metrics(metrics_file):
    """Print system metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)

        latest = data.get('latest', {})
        stats = data.get('stats', {})

        print("=" * 70)
        print("[System Resources Stats]")
        print("=" * 70)
        print("  CPU Avg:    " + str(round(stats.get('cpu_avg', 0), 1)) + "%")
        print("  CPU Max:    " + str(round(stats.get('cpu_max', 0), 1)) + "%")
        print("  RAM Avg:    " + str(round(stats.get('ram_avg', 0), 1)) + "%")
        print("  RAM Max:    " + str(round(stats.get('ram_max', 0), 1)) + "%")
        if 'gpu_avg' in stats:
            print("  GPU Avg:    " + str(round(stats.get('gpu_avg', 0), 1)) + "%")
            print("  GPU Max:    " + str(round(stats.get('gpu_max', 0), 1)) + "%")
        print("  Samples:    " + str(data.get('sample_count', 0)))

        if 'training_cpu_avg' in stats or 'battle_cpu_avg' in stats:
            print()
            print("[Phase Stats]")
            print("=" * 70)
            if 'training_cpu_avg' in stats:
                print("  Training CPU Avg: " + str(round(stats.get('training_cpu_avg', 0), 1)) + "%")
                print("  Training GPU Avg: " + str(round(stats.get('training_gpu_avg', 0), 1)) + "%")
            if 'battle_cpu_avg' in stats:
                print("  Battle CPU Avg:   " + str(round(stats.get('battle_cpu_avg', 0), 1)) + "%")
                print("  Battle GPU Avg:   " + str(round(stats.get('battle_gpu_avg', 0), 1)) + "%")

        if latest:
            print()
            print("[Latest Sample]")
            print("=" * 70)
            print("  Timestamp: " + str(latest.get('timestamp', '')))
            print("  Phase:     " + str(latest.get('phase', '')))
            print("  CPU:       " + str(round(latest.get('cpu_percent', 0), 1)) + "%")
            ram_used = round(latest.get('ram_used_gb', 0), 1)
            ram_total = round(latest.get('ram_total_gb', 0), 1)
            print("  RAM:       " + str(round(latest.get('ram_percent', 0), 1)) + "% (" + str(ram_used) + "GB / " + str(ram_total) + "GB)")
            if 'gpu_util' in latest:
                print("  GPU Util:   " + str(latest.get('gpu_util', 0)) + "%")
                print("  GPU Mem:   " + str(latest.get('gpu_mem_used_mb', 0)) + "MB / " + str(latest.get('gpu_mem_total_mb', 0)) + "MB")
                print("  GPU Temp:  " + str(latest.get('gpu_temp', 0)) + "C")

        print("=" * 70)

    except Exception as e:
        print("Error: " + str(e))
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python print_metrics.py <type> <file>")
        print("  type: training|system")
        sys.exit(1)

    metric_type = sys.argv[1]
    metric_file = sys.argv[2]

    if metric_type == 'training':
        print_training_metrics(metric_file)
    elif metric_type == 'system':
        print_system_metrics(metric_file)
    else:
        print("Unknown type: " + metric_type)
        sys.exit(1)
