"""统计亚军动作组的「op 类型序列」模板：2-op / 3-op 到底由哪些类别组合构成。

用途：如果要枚举 2-op 候选，只需要枚举**高频模板** × 位置网裁剪后的位置，
而不是 40×40 的笛卡尔积。
"""
import collections
import glob
import json
import sys

NAMES = {11: "建", 12: "升", 13: "降", 21: "闪", 31: "速", 32: "蚁"}


def parse_key(key: str):
    if key == "hold":
        return []
    ops = []
    for segment in key.split("|"):
        if segment.startswith("F:"):
            continue
        parts = segment.split(":")
        if len(parts) < 2:
            continue
        ops.append(int(parts[0]))
    return ops


files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else
                         "match_results/ai0_runnerup_seed73*.stderr.log"))[:8]
templates = {2: collections.Counter(), 3: collections.Counter()}
lengths = collections.Counter()
total = 0
for path in files:
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"rank":1,' not in line or "base_heuristic" not in line:
                continue
            record = json.loads(line)
            ops = parse_key(record["key"])
            total += 1
            lengths[len(ops)] += 1
            if len(ops) in (2, 3):
                templates[len(ops)][tuple(ops)] += 1

print(f"[日志] {len(files)} 局，{total} 个决策")
print("动作组长度分布：", dict(sorted(lengths.items())))
for n in (2, 3):
    counter = templates[n]
    subtotal = sum(counter.values())
    if not subtotal:
        continue
    print(f"\n=== {n}-op 动作组（共 {subtotal} 条，占全部 {100*subtotal/total:.1f}%）===")
    cumulative = 0
    for index, (template, count) in enumerate(counter.most_common(12)):
        cumulative += count
        name = " + ".join(NAMES.get(t, str(t)) for t in template)
        print(f"  {name:<12} {count:>5} 条  {100*count/subtotal:5.1f}%   "
              f"（累计 {100*cumulative/subtotal:5.1f}%）")
    print(f"  不同模板数 = {len(counter)}")
