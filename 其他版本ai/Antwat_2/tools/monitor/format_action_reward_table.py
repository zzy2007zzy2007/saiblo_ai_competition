#!/usr/bin/env python3
"""
从 episode_batch_battle_stats.jsonl 读取 EpisodeBatch 级 Action Reward 统计表。

每行对应一个 EpisodeBatch 聚合窗口（非累积），展示各 reward 分量的占比和汇总。

占比计算（需求第8条）：
  - abs_total = sum(|各分量|)  所有分量绝对值累加的总量
  - 各分量占比 = 分量值 / abs_total * 100  （保留正负方向）
  - Reward = 该窗口平均总奖励 (avg_reward)
  - PosRatio = 正分量之和 / abs_total

用法:
  format_action_reward_table.py --battle-stats outputs/run_id/training/episode_batch_battle_stats.jsonl

可选:
  --interval    输出间隔（默认=1，即每行都输出）
"""
import json
import sys
import os


# 动作奖励分量（9个）
ACTION_COMPONENTS = [
    ('noop', 'Noop'),
    ('build_tower', 'Build'),
    ('upgrade_tower', 'Upgrd'),
    ('downgrade_tower', 'Downg'),
    ('lightning_storm', 'LS'),
    ('emp_blaster', 'Emp'),
    ('deflector', 'Defl'),
    ('evasion', 'Eva'),
    ('tech_upgrade', 'Tech'),
]

# 非动作奖励分量（8个）
NON_ACTION_COMPONENTS = [
    ('avg_rw_hp_attack_base', 'HPDmg'),
    ('avg_rw_hp_attack_tower', 'TwrDmg'),
    ('avg_rw_coin_gain',     'Coin'),
    ('avg_rw_tower_survival', 'TwrSv'),
    ('avg_rw_balance',       'Bal'),
    ('avg_rw_tech_bonus',    'TechB'),
    ('avg_rw_die_penalty',   'DieP'),
    ('avg_rw_end_reward',    'EndR'),
]


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

    # 列定义: (显示名, 类型, 数据键或None)
    # 类型: 'action_pct' / 'non_action_val' / 'summary'
    col_defs = []
    # Epid 列
    col_defs.append(('Epid', 'epid', None))
    # 9个动作奖励占比
    for key, label in ACTION_COMPONENTS:
        col_defs.append((label, 'action_pct', key))
    # 8个非动作奖励值
    for key, label in NON_ACTION_COMPONENTS:
        col_defs.append((label, 'non_action_val', key))
    # 汇总列
    col_defs.append(('Rounds',   'summary', 'avg_rounds'))
    col_defs.append(('Reward',   'summary', 'avg_reward'))
    col_defs.append(('MaxBal',   'summary', 'avg_max_own_coins'))
    col_defs.append(('TotInc',   'summary', 'avg_total_own_coin_income'))
    col_defs.append(('AbsTotal', 'summary', '__abs_total__'))
    col_defs.append(('PosRatio', 'summary', '__pos_ratio__'))

    # 计算列宽
    col_widths = []
    for name, ctype, _ in col_defs:
        if ctype == 'epid':
            w = 6
        elif ctype in ('action_pct', 'non_action_val'):
            w = max(len(name) + 1, 7)
        elif name in ('Reward', 'AbsTotal'):
            w = 9
        else:
            w = max(len(name) + 1, 7)
        col_widths.append(w)

    header_parts = [f"{n:>{w}}" for n, w in
                    zip([c[0] for c in col_defs], col_widths)]
    header = "  ".join(header_parts)
    sep = "  ".join('-' * w for w in col_widths)

    print("Action Reward 统计表（EpisodeBatch 聚合，每局平均):")
    print("")
    print(header)
    print(sep)

    output_count = 0
    for idx, r in enumerate(records):
        if (idx + 1) % interval != 0 and idx != len(records) - 1:
            continue

        ep_end = r.get('episode_end', 0)
        avg_ar = r.get('avg_action_rewards', {})

        # 收集所有分量值用于 abs_total 计算
        comp_values = {}
        for key, label in ACTION_COMPONENTS:
            comp_values[label] = avg_ar.get(key, 0.0)
        for key, label in NON_ACTION_COMPONENTS:
            comp_values[label] = r.get(key, 0.0)

        abs_total = sum(abs(v) for v in comp_values.values())
        pos_sum = sum(v for v in comp_values.values() if v > 0)

        row_str = f"E{ep_end:>5d}"

        for (name, ctype, key), w in zip(col_defs[1:], col_widths[1:]):
            if ctype == 'action_pct':
                val = comp_values.get(name, 0.0)
                if abs_total > 0:
                    pct = val / abs_total * 100
                else:
                    pct = 0.0
                row_str += f"  {pct:>{w}.1f}"

            elif ctype == 'non_action_val':
                # 非动作奖励也用比例展示，与需求第8条一致
                val = comp_values.get(name, 0.0)
                if abs_total > 0:
                    pct = val / abs_total * 100
                else:
                    pct = 0.0
                row_str += f"  {pct:>{w}.1f}"

            elif ctype == 'summary':
                if key == '__abs_total__':
                    row_str += f"  {abs_total:>{w}.1f}"
                elif key == '__pos_ratio__':
                    if abs_total > 0:
                        pr = pos_sum / abs_total * 100
                    else:
                        pr = 0.0
                    row_str += f"  {pr:>{w}.1f}"
                else:
                    val = r.get(key, 0.0)
                    if name == 'Reward':
                        row_str += f"  {val:>{w}.2f}"
                    else:
                        row_str += f"  {val:>{w}.1f}"

        print(row_str)
        output_count += 1

    if output_count == 0:
        print("(无数据)")
        return

    print()
    print(f"Total: {len(records)} EpisodeBatch windows")
    print("--- 列说明 ---")
    print("Epid        = Episode Batch ID (E20 = ep 1~20 聚合)")
    print("Noop~Tech   = 各 action type 的 reward 占比 (%)")
    print("HPDmg~EndR  = 非动作 reward 分量占比 (%)")
    print("Rounds      = 该窗口平均每局回合数")
    print("Reward      = 该窗口平均每局总奖励")
    print("MaxBal      = 该窗口平均每局最大金币余额")
    print("TotInc      = 该窗口平均每局金币累计收入")
    print("AbsTotal    = 各分量绝对值累加总量 (每局平均)")
    print("PosRatio    = 正分量之和 / AbsTotal (%)")
    print("公式: 分量占比 = 分量值 / sum(|各分量|) * 100 （保留正负号）")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Format reward component table (EpisodeBatch aggregated)')
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
