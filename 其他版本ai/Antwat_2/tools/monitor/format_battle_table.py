#!/usr/bin/env python3
"""
从 eval_battle_log.jsonl + episode_batch_baseline_winrates.jsonl 生成 Baseline 对战横表。

输入:
  --eval-log   eval_battle_log.jsonl              ← 每局原始数据
  --winrates   episode_batch_baseline_winrates.jsonl  ← 聚合胜率

输出: 按 EpisodeBatch 窗口和 Baseline Agent 分组的胜率横表

用法:
  format_battle_table.py \
    --eval-log outputs/run_id/evaluation/eval_battle_log.jsonl \
    --winrates outputs/run_id/evaluation/episode_batch_baseline_winrates.jsonl
"""
import json
import os
import sys
from collections import OrderedDict


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


def format_battle_table(eval_log, winrates, interval=1, n_envs=20):
    # --- 1. 解析聚合胜率数据（episode_batch_baseline_winrates.jsonl）---
    # 取最后一条，获取 all baseline agents
    winrates_data = winrates[-1] if winrates else {}
    baseline_winrates = winrates_data.get("baseline_winrates", {})
    baseline_names = list(baseline_winrates.keys())

    # --- 2. 解析原始 eval_battle_log.jsonl ---
    # 按 episode 分组，再按 baseline_agent_name 分组
    eval_by_ep = OrderedDict()  # episode -> {agent_name: [entries]}
    for entry in eval_log:
        ep = entry.get('episode', 0)
        agent = entry.get('baseline_agent_name', 'unknown')
        if ep not in eval_by_ep:
            eval_by_ep[ep] = {}
        if agent not in eval_by_ep[ep]:
            eval_by_ep[ep][agent] = []
        eval_by_ep[ep][agent].append(entry)

    # 从 eval_log 中收集所有出现的 baseline agent 名
    eval_baseline_names = set()
    for ep_data in eval_by_ep.values():
        eval_baseline_names.update(ep_data.keys())

    # 合并：优先用 winrates 中的 agents，不足的用 eval_log 中的补齐
    all_baseline_names = list(OrderedDict.fromkeys(baseline_names + list(eval_baseline_names)))
    if not all_baseline_names:
        all_baseline_names = list(eval_baseline_names)

    if not all_baseline_names:
        print("(无对战数据 — eval_battle_log.jsonl 为空或不存在)")
        return

    col_width = 18
    header = f"{'EpBatch':<10}"
    for name in all_baseline_names:
        header += f"  {name[:col_width]:<{col_width}}"
    header += f"  {'WinRate':<8}  {'AvgRds':<8}"

    print()
    sep = "=" * len(header)
    print(sep)
    print(f"Baseline Battle Stats (n_envs={n_envs}, interval={interval})")
    print(sep)
    print(header)
    print("-" * len(header))

    # 全局累积
    total_wins = {name: 0 for name in all_baseline_names}
    total_battles = {name: 0 for name in all_baseline_names}
    total_rounds = {name: 0 for name in all_baseline_names}

    # 从 winrates 数据中按窗口输出
    for t_idx, wr_entry in enumerate(winrates):
        # 按 interval 采样，始终显示最后一条
        if (t_idx + 1) % interval != 0 and t_idx != len(winrates) - 1:
            continue

        ep_start = wr_entry.get('episode_start', '?')
        ep_end = wr_entry.get('episode_end', '?')
        baselines = wr_entry.get('baseline_winrates', {})

        row = f"E{ep_start}~E{ep_end:<1}"
        rw, rt = 0, 0
        for name in all_baseline_names:
            stats = baselines.get(name, {})
            wins = stats.get('wins', 0)
            total = stats.get('battles', 0)
            avg_rds = stats.get('avg_rounds', 0)
            rw += wins
            rt += total
            total_wins[name] += wins
            total_battles[name] += total
            total_rounds[name] += avg_rds * total
            if total > 0:
                cell = f"{wins/total*100:.0f}%({wins}/{total})"
                row += f"  {cell:<{col_width}}"
            else:
                row += f"  {'N/A':<{col_width}}"
        wr = rw / rt * 100 if rt else 0
        row += f"  {wr:>5.0f}%"

        # 平均回合数
        total_rd_sum = sum(
            stats.get('avg_rounds', 0) * stats.get('battles', 0)
            for stats in baselines.values()
        )
        total_bt = sum(
            stats.get('battles', 0) for stats in baselines.values()
        )
        row += f"  {total_rd_sum / max(total_bt, 1):>7.1f}"
        print(row)

    # 如果没有 winrates 数据，尝试从 eval_log 原始数据构造
    if not winrates and eval_log:
        # 按 episode 分组显示
        for ep, ep_data in eval_by_ep.items():
            row = f"E{ep:<7}"
            rw, rt = 0, 0
            for name in all_baseline_names:
                entries = ep_data.get(name, [])
                total = len(entries)
                wins = sum(1 for e in entries if e.get('result') == 'win')
                rw += wins
                rt += total
                total_wins[name] += wins
                total_battles[name] += total
                total_rounds[name] += sum(e.get('rounds', 0) for e in entries)
                if total > 0:
                    cell = f"{wins/total*100:.0f}%({wins}/{total})"
                    row += f"  {cell:<{col_width}}"
                else:
                    row += f"  {'N/A':<{col_width}}"
            wr = rw / rt * 100 if rt else 0
            row += f"  {wr:>5.0f}%"
            total_rd = sum(e.get('rounds', 0) for entries in ep_data.values() for e in entries)
            total_bt = sum(len(entries) for entries in ep_data.values())
            row += f"  {total_rd / max(total_bt, 1):>7.1f}"
            print(row)

    print("-" * len(header))

    # 总计行
    total_row = f"{'Total':<10}"
    tw, tt = 0, 0
    for name in all_baseline_names:
        wins = total_wins[name]
        total = total_battles[name]
        tw += wins
        tt += total
        if total > 0:
            cell = f"{wins/total*100:.0f}%({wins}/{total})"
            total_row += f"  {cell:<{col_width}}"
        else:
            total_row += f"  {'N/A':<{col_width}}"
    overall_wr = tw / tt * 100 if tt else 0
    total_avg_rd = sum(total_rounds.values()) / max(tt, 1)
    total_row += f"  {overall_wr:>5.0f}%  {total_avg_rd:>7.1f}"
    print(total_row)
    print(sep)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Format baseline battle results as cross-table')
    parser.add_argument('--eval-log', '-e', type=str, default=None,
                        help='Path to eval_battle_log.jsonl')
    parser.add_argument('--winrates', '-w', type=str, default=None,
                        help='Path to episode_batch_baseline_winrates.jsonl')
    parser.add_argument('--interval', type=int, default=1,
                        help='Show every N windows (default: 1 = every n_envs period)')
    parser.add_argument('--n-envs', type=int, default=20,
                        help='Number of parallel envs (for display metadata)')
    args = parser.parse_args()

    if not args.eval_log and not args.winrates:
        print("错误: 至少需要 --eval-log 或 --winrates 之一", file=sys.stderr)
        sys.exit(1)

    eval_log = load_jsonl(args.eval_log)
    winrates = load_jsonl(args.winrates)

    format_battle_table(eval_log, winrates, interval=args.interval, n_envs=args.n_envs)


if __name__ == '__main__':
    main()
