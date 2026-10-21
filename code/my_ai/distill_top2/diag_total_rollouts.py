"""看几个决策：日志里的 total_rollouts 与各 plan 行 rollouts 之和到底差在哪。"""
import json

PATH = "match_results/ai0_runnerup_seed732.stderr.log"
arms = []
shown = 0
for line in open(PATH, encoding="utf-8", errors="replace"):
    if '"kind":"decision"' in line:
        record = json.loads(line)
        total = record["total_rollouts"]
        summed = sum(arms)
        print(f"决策 r={record['round']:>3} p={record['player']}: "
              f"plans_total={record['plans_total']:>4} 日志total={total:>6} "
              f"各plan之和={summed:>6} 条数={len(arms):>4} 比值={summed / max(total, 1):.3f}")
        arms = []
        shown += 1
        if shown >= 8:
            break
    elif "base_heuristic" in line:
        arms.append(json.loads(line)["rollouts"])
