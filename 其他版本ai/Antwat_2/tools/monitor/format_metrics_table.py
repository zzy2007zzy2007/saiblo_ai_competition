#!/usr/bin/env python3
"""
从 episode_batch_train_stats.jsonl + episode_batch_battle_stats.jsonl 读取 EpisodeBatch 级训练指标汇总表。

每行对应一个 EpisodeBatch 聚合窗口（非累积），展示该窗口内的平均训练技术指标 + 业务指标。

用法:
  format_metrics_table.py \
    --train-stats outputs/run_id/training/episode_batch_train_stats.jsonl \
    --battle-stats outputs/run_id/training/episode_batch_battle_stats.jsonl

可选:
  --interval    输出间隔（默认=1，即每行都输出）
"""
import json
import sys
import os


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


def format_table(train_stats, battle_stats, interval=1):
    if not train_stats:
        print("(无 episode_batch_train_stats 数据 — 等待第一个 EpisodeBatch 聚合完成)")
        return

    # 按 episode_end 建立 battle_stats 索引
    battle_by_end = {}
    for b in battle_stats:
        battle_by_end[b.get('episode_end')] = b

    # 列定义: (显示名, 字段来源, 数据键, 格式)
    # 字段来源: 'train' = train_stats, 'battle' = battle_stats
    cols = [
        ('Epid',      None,       None,                         'id'),
        ('Rounds',    'battle',   'avg_rounds',                 '.4g'),
        ('Reward',    'battle',   'avg_reward',                 '.4g'),
        ('MaxBal',    'battle',   'avg_max_own_coins',          '.4g'),
        ('TotInc',    'battle',   'avg_total_own_coin_income',  '.4g'),
        ('WinRate',   'battle',   'win_rate',                   '.0%'),
        ('PolicyLoss','train',    'avg_policy_loss',            '.4g'),
        ('VLoss',     'train',    'avg_value_loss',             '.4g'),
        ('AuxTwrL',   'train',    'avg_aux_tower_loss',         '.4g'),
        ('AuxGldL',   'train',    'avg_aux_gold_loss',          '.4g'),
        ('AuxBaseL',  'train',    'avg_aux_base_loss',          '.4g'),
        ('AuxETwrL',  'train',    'avg_aux_enemy_tower_loss',   '.4g'),
        ('AuxEGldL',  'train',    'avg_aux_enemy_gold_loss',    '.4g'),
        ('AuxEBaseL', 'train',    'avg_aux_enemy_base_loss',    '.4g'),
        ('Entropy',   'train',    'avg_entropy',                '.4g'),
        ('ClipFrac',  'train',    'avg_clip_fraction',          '.3g'),
        ('GradNorm',  'train',    'avg_grad_norm',              '.4g'),
        ('LR',        'train',    'avg_learning_rate',          '.4g'),
    ]

    # 计算每列宽度 + 表头
    col_widths = []
    header_parts = []
    for name, src, key, fmt in cols:
        if name == 'Epid':
            w = 6
        elif fmt == '.0%':
            w = 8
        else:
            min_w = max(len(name) + 2, 9)
            w = min_w
        col_widths.append(w)
        header_parts.append(f"{name:>{w}}")
    header = "  ".join(header_parts)
    sep = "  ".join('-' * w for w in col_widths)

    print("训练指标表（EpisodeBatch 聚合，非累积):")
    print("")
    print(header)
    print(sep)

    output_count = 0
    for t_idx, t in enumerate(train_stats):
        if (t_idx + 1) % interval != 0 and t_idx != len(train_stats) - 1:
            continue

        ep_end = t.get('episode_end', 0)
        b = battle_by_end.get(ep_end, {})

        row_parts = []
        for name, src, key, fmt in cols:
            w = col_widths[cols.index((name, src, key, fmt))]
            if name == 'Epid':
                row_parts.append(f"E{ep_end:>{w-1}d}")
                continue
            if src == 'train':
                val = t.get(key)
            else:
                val = b.get(key)

            if val is None:
                row_parts.append('N/A'.rjust(w))
            elif fmt == '.0%':
                row_parts.append(f"{val:.0%}".rjust(w))
            else:
                row_parts.append(f"{val:{fmt}}".rjust(w))

        print("  ".join(row_parts))
        output_count += 1

    if output_count == 0:
        print("(无训练历史数据)")
        return

    print()
    print(f"Total: {len(train_stats)} EpisodeBatch windows")
    if battle_stats:
        print(f"对战聚合: {len(battle_stats)} 窗口")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Format training metrics table (EpisodeBatch aggregated)')
    parser.add_argument('--train-stats', type=str, required=True,
                        help='Path to episode_batch_train_stats.jsonl')
    parser.add_argument('--battle-stats', type=str, required=True,
                        help='Path to episode_batch_battle_stats.jsonl')
    parser.add_argument('--interval', '-i', type=int, default=1,
                        help='Output interval (default: 1 = show all)')
    args = parser.parse_args()

    train_stats = load_jsonl(args.train_stats)
    battle_stats = load_jsonl(args.battle_stats)

    if not train_stats:
        print("(无训练统计数据 — 文件不存在或为空)")
        print(f"  train_stats: {args.train_stats}")
        sys.exit(0)

    format_table(train_stats, battle_stats, args.interval)


if __name__ == '__main__':
    main()
