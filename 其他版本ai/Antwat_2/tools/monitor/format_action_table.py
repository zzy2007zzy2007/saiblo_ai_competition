#!/usr/bin/env python3
"""
从 episode_batch_battle_stats.jsonl 读取 EpisodeBatch 级动作次数统计表。

每行对应一个 EpisodeBatch 聚合窗口（非累积），展示该窗口内各动作类型的平均每局执行次数。

用法:
  format_action_table.py --battle-stats outputs/run_id/training/episode_batch_battle_stats.jsonl

可选:
  --interval    输出间隔（默认=1，即每行都输出）
"""
import json
import sys
import os


ACTION_KEYS = [
    'noop', 'build_tower', 'upgrade_tower', 'downgrade_tower',
    'lightning_storm', 'emp_blaster', 'deflector', 'evasion', 'tech_upgrade',
]
ACTION_LABELS = ['Noop', 'Build', 'Upgrd', 'Downg', 'LS', 'Emp', 'Defl', 'Eva', 'Tech']


def load_jsonl(path):
    if not path or not os.path.isfile(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_table(records, interval=1):
    if not records:
        print("(无 episode_batch_battle_stats 数据)")
        return

    # 列宽
    col_names = ['Epid'] + ACTION_LABELS + ['Total', 'Rounds', 'Reward', 'MaxBal', 'TotInc']
    col_widths = []
    for name in col_names:
        if name == 'Epid':
            w = 6
        else:
            w = max(len(name) + 1, 7)
        col_widths.append(w)

    header_parts = [f"{n:>{w}}" for n, w in zip(col_names, col_widths)]
    header = "  ".join(header_parts)
    sep = "  ".join('-' * w for w in col_widths)

    print("Action 执行次数表（EpisodeBatch 聚合，每局平均):")
    print("")
    print(header)
    print(sep)

    output_count = 0
    for idx, r in enumerate(records):
        if (idx + 1) % interval != 0 and idx != len(records) - 1:
            continue

        ep_end = r.get('episode_end', 0)
        avg_ac = r.get('avg_action_counts', {})
        avg_rounds = r.get('avg_rounds', 0)
        avg_reward = r.get('avg_reward', 0.0)
        avg_max_bal = r.get('avg_max_own_coins', 0.0)
        avg_tot_inc = r.get('avg_total_own_coin_income', 0.0)

        row_vals = []
        total_count = 0.0
        for k in ACTION_KEYS:
            v = avg_ac.get(k, 0.0)
            row_vals.append(v)
            total_count += v

        # Epid 列
        row_str = f"E{ep_end:>5d}"
        # 动作列
        for v, w in zip(row_vals, col_widths[1:1+len(ACTION_KEYS)]):
            row_str += f"  {v:>{w}.2f}"
        # Total 列
        row_str += f"  {total_count:>{col_widths[-5]}.2f}"
        # Rounds 列
        row_str += f"  {avg_rounds:>{col_widths[-4]}.1f}"
        # Reward 列
        row_str += f"  {avg_reward:>{col_widths[-3]}.2f}"
        # MaxBal 列
        row_str += f"  {avg_max_bal:>{col_widths[-2]}.1f}"
        # TotInc 列
        row_str += f"  {avg_tot_inc:>{col_widths[-1]}.2f}"

        print(row_str)
        output_count += 1

    if output_count == 0:
        print("(无数据)")
        return

    print()
    print(f"Total: {len(records)} EpisodeBatch windows")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Format action count table (EpisodeBatch aggregated)')
    parser.add_argument('--battle-stats', type=str, required=True,
                        help='Path to episode_batch_battle_stats.jsonl')
    parser.add_argument('--interval', '-i', type=int, default=1,
                        help='Output interval (default: 1 = show all)')
    args = parser.parse_args()

    records = load_jsonl(args.battle_stats)
    if not records:
        print("(无对战统计数据 — 文件不存在或为空)")
        print(f"  battle_stats: {args.battle_stats}")
        sys.exit(0)

    format_table(records, args.interval)


if __name__ == '__main__':
    main()
