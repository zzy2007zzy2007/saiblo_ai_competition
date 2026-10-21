"""为「用神经网络生成候选动作」这个目标做落地位分析。

关键问题（决定网络输出空间怎么设计）：
1. 亚军实际选中的动作是什么？（hold / 建塔 / 升级 / 降级 / 闪电）
2. 选中计划的 op 数分布（生成器要吐 1 个 op 还是 op 序列？）
3. 选中动作的**位置**有多集中？（建塔只有 15 个 code；闪电是一个半径内的一批格子）
4. 用"类别 × 位置"这种因子化生成器时，top-K 位置能覆盖多少选中动作 ⇒ recall@K 的上界

数据源：亚军自己的 plans 日志（rank==1 就是它实际下发的计划）。
"""
from __future__ import annotations

import collections
import json
import statistics
import sys

OP_NAMES = {11: "build", 12: "upgrade", 13: "downgrade", 21: "lightning",
            31: "gen_speed", 32: "ant_level"}


def parse_key(key: str) -> tuple[list[tuple[int, int, int]], int]:
    """key -> (ops, followup 步数)。hold 返回 ([], 0)。"""
    if key == "hold":
        return [], 0
    ops, followup = [], 0
    for segment in key.split("|"):
        if segment.startswith("F:"):
            followup += 1
            continue
        parts = segment.split(":")
        if len(parts) < 2:
            continue
        try:
            values = [int(x) for x in parts[:3]]
        except ValueError:
            continue
        while len(values) < 3:
            values.append(-1)
        ops.append((values[0], values[1], values[2]))
    return ops, followup


def main() -> int:
    n_ops_counter = collections.Counter()
    first_type = collections.Counter()
    all_types = collections.Counter()
    followup_counter = collections.Counter()
    build_pos = collections.Counter()
    lightning_pos = collections.Counter()
    first_op = collections.Counter()
    total = 0
    hold = 0
    decisions = 0

    for path in sys.argv[1:]:
        print(f"[读] {path.split(chr(92))[-1]}")
        for line in open(path, encoding="utf-8", errors="replace"):
            if "base_heuristic" not in line:
                continue
            record = json.loads(line)
            if record.get("rank") != 1:
                continue
            decisions += 1
            key = record["key"]
            ops, followup = parse_key(key)
            n_ops_counter[len(ops)] += 1
            followup_counter[followup] += 1
            if not ops:
                hold += 1
                continue
            total += 1
            first_type[ops[0][0]] += 1
            for op in ops:
                all_types[op[0]] += 1
            first_op[(ops[0][0], ops[0][1], ops[0][2])] += 1
            for op_type, arg0, arg1 in ops:
                if op_type == 11:
                    build_pos[(arg0, arg1)] += 1
                elif op_type == 21:
                    lightning_pos[(arg0, arg1)] += 1

    print(f"\n=== {decisions} 个决策 ===")
    print(f"不动(hold): {hold} ({100.0 * hold / decisions:.1f}%)   有动作: {total} ({100.0 * total / decisions:.1f}%)")
    print(f"\n选中计划的 op 数分布: {sorted(n_ops_counter.items())}")
    print(f"followup 步数分布: {sorted(followup_counter.items())}")
    print(f"\n首个 op 类型: " + ", ".join(
        f"{OP_NAMES.get(k, k)}:{v}({100.0 * v / total:.1f}%)" for k, v in first_type.most_common()))
    print("所有 op 类型: " + ", ".join(
        f"{OP_NAMES.get(k, k)}:{v}" for k, v in all_types.most_common()))

    def coverage(counter: collections.Counter, label: str) -> None:
        values = sorted(counter.values(), reverse=True)
        summed = sum(values)
        print(f"\n{label}: 出现 {len(counter)} 种位置，总 {summed} 次")
        for k in (1, 4, 8, 16, 32, 64, 128):
            if k > len(values):
                k = len(values)
            print(f"  top-{k:<4} 位置覆盖 {100.0 * sum(values[:k]) / summed:5.1f}%", end="")
            if k >= len(values):
                print("  (全部)")
                break
            print()

    coverage(build_pos, "建塔位置(code)")
    coverage(lightning_pos, "闪电位置")
    print("\n建塔位置 top-12:", build_pos.most_common(12))
    print("闪电位置 top-12:", lightning_pos.most_common(12))
    print(f"\n首个 op 的 (类型,arg0,arg1) 组合数: {len(first_op)}")
    print("最常见首个 op:", first_op.most_common(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
