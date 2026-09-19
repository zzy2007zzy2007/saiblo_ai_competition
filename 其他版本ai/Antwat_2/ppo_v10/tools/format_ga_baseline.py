#!/usr/bin/env python3
"""GA 基线验证结果查看工具。

读取 outputs/<run_id>/generations/gen_*/evaluation.json，
展示最新一代的基线胜率明细和最近 N 代的胜率趋势。

用法:
    python tools/format_ga_baseline.py --run-dir outputs/20260612_120000 -g 5
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_evaluation(run_dir: str) -> dict[int, dict]:
    """加载所有代的 evaluation.json。

    Returns:
        {gen_num: {seed_id: {baseline_name: {agent1_wins, agent2_wins, draws, total_battles}}}}
    """
    generations = {}
    gen_dir_pattern = os.path.join(run_dir, "generations")
    if not os.path.isdir(gen_dir_pattern):
        return generations

    for entry in sorted(os.listdir(gen_dir_pattern)):
        match = entry.startswith("gen_") and entry[4:].isdigit()
        if not match:
            continue

        gen_num = int(entry[4:])
        eval_path = os.path.join(gen_dir_pattern, entry, "evaluation.json")
        if not os.path.isfile(eval_path):
            continue

        with open(eval_path, "r") as f:
            data = json.load(f)
        generations[gen_num] = data

    return generations


def format_table(headers: list[str], rows: list[list[str]], col_widths: list[int] | None = None) -> str:
    """格式化对齐表格。"""
    if col_widths is None:
        col_widths = [
            max(len(str(row[i])) for row in [headers] + rows)
            for i in range(len(headers))
        ]
        col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]

    def fmt_row(cells):
        return "| " + " | ".join(
            str(c).ljust(w) if i == 0 else str(c).rjust(w)
            for i, (c, w) in enumerate(zip(cells, col_widths))
        ) + " |"

    lines = [fmt_row(headers)]
    lines.append("|-" + "-|-".join("-" * w for w in col_widths) + "-|")
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def get_best_seed_results(eval_data: dict) -> dict[str, dict]:
    """从一代的 evaluation.json 中提取最佳种子的各 baseline 结果。

    最佳种子 = 对最强 baseline 胜率最高的种子。
    """
    if not eval_data:
        return {}

    # 找出所有 baseline agent 名称
    baseline_names = set()
    for seed_results in eval_data.values():
        baseline_names.update(seed_results.keys())
    if not baseline_names:
        return {}

    # 排序 baseline：取最难的（历史上种子胜率最低的）作为决胜 baseline
    # 默认用第一个 baseline
    decisive_baseline = sorted(baseline_names)[0]

    best_seed_id = None
    best_wr = -1.0

    for seed_id, seed_results in eval_data.items():
        if decisive_baseline not in seed_results:
            continue
        br = seed_results[decisive_baseline]
        wr = br.get("agent1_wins", 0) / max(br.get("total_battles", 1), 1)
        if wr > best_wr:
            best_wr = wr
            best_seed_id = seed_id

    if best_seed_id is None:
        return {}

    return eval_data[best_seed_id]


def print_latest_gen(eval_data: dict, gen: int):
    """打印最新一代的基线胜率明细。"""
    print(f"\n{'=' * 60}")
    print(f"  基线验证结果 — Gen {gen}")
    print(f"{'=' * 60}")

    if not eval_data:
        print("  (无数据)")
        return

    best_results = get_best_seed_results(eval_data)
    if not best_results:
        print("  (无有效 baseline 数据)")
        return

    rows = []
    for name, br in sorted(best_results.items()):
        w = br.get("agent1_wins", 0)
        l = br.get("agent2_wins", 0)
        d = br.get("draws", 0)
        total = br.get("total_battles", 0)
        wr = w / max(total, 1) * 100
        rows.append([name, str(w), str(l), str(d), str(total), f"{wr:.1f}%"])

    print()
    print(format_table(
        ["Baseline", "胜", "负", "平", "场次", "胜率"],
        rows,
    ))


def print_trend(generations: dict[int, dict], last_n: int):
    """打印最近 N 代的胜率趋势。"""
    if not generations:
        return

    sorted_gens = sorted(generations.keys())
    recent_gens = sorted_gens[-last_n:]

    # 收集所有出现的 baseline
    all_baselines = set()
    for gen_data in generations.values():
        for seed_results in gen_data.values():
            all_baselines.update(seed_results.keys())
    all_baselines = sorted(all_baselines)

    print(f"\n{'=' * 60}")
    print(f"  胜率趋势（最近 {last_n} 代）")
    print(f"{'=' * 60}\n")

    # 每代取 best seed 的胜率
    headers = ["Baseline"] + [f"Gen{g}" for g in recent_gens] + ["趋势"]
    rows = []

    for bl_name in all_baselines:
        row = [bl_name]
        winrates = []
        for g in recent_gens:
            gen_data = generations.get(g, {})
            best = get_best_seed_results(gen_data)
            bl_result = best.get(bl_name, {})
            wr = bl_result.get("agent1_wins", 0) / max(bl_result.get("total_battles", 1), 1) * 100
            winrates.append(wr)
            row.append(f"{wr:.0f}%")
        # 趋势箭头
        if len(winrates) >= 2:
            delta = winrates[-1] - winrates[0]
            if abs(delta) < 2:
                row.append("→")
            elif delta > 0:
                row.append("↗")
            else:
                row.append("↘")
        else:
            row.append("—")
        rows.append(row)

    print(format_table(headers, rows))


def main():
    parser = argparse.ArgumentParser(description="GA 基线验证结果查看工具")
    parser.add_argument("--run-dir", required=True, help="outputs/<run_id> 目录路径")
    parser.add_argument("-g", type=int, default=5, help="趋势图显示最近 N 代（默认 5）")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"错误: 目录不存在 — {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    generations = load_evaluation(args.run_dir)
    if not generations:
        print("未找到任何 evaluation.json 文件。", file=sys.stderr)
        sys.exit(1)

    sorted_gens = sorted(generations.keys())
    latest_gen = sorted_gens[-1]

    print_latest_gen(generations[latest_gen], latest_gen)
    print_trend(generations, args.g)


if __name__ == "__main__":
    main()
