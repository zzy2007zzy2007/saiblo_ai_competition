"""确认亚军一次决策到底跑多少次 rollout（多局统计）。

日志里每个决策有一行 `{"kind":"decision", ..., "total_rollouts": N, "plans_total": M}`，
后面跟着 M 条 plan 行，每条带自己的 `rollouts`。这里同时校验 "总条数 == 各 plan 之和"。
"""
import glob
import json
import statistics
import sys

files = sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else
                         "match_results/ai0_runnerup_seed7*.stderr.log"))[:10]
print(f"[日志] {len(files)} 个")

all_totals, all_plans, all_arms = [], [], []
mismatch = 0
decisions = 0
for path in files:
    # ⚠️ plan 行在 decision 行**之前**打印 ⇒ 先攒 arms，遇到 decision 行再结账
    arms = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"kind":"decision"' in line:
                record = json.loads(line)
                decisions += 1
                all_totals.append(record["total_rollouts"])
                all_plans.append(record["plans_total"])
                all_arms.extend(arms)
                if sum(arms) != record["total_rollouts"]:
                    mismatch += 1
                arms = []
            elif "base_heuristic" in line:
                arms.append(json.loads(line)["rollouts"])
ordered = sorted(all_totals)


def pct(p):
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


print(f"\n=== {decisions} 个决策 ===")
print(f"每决策 rollout 总数：中位 {statistics.median(ordered):.0f}  "
      f"均值 {statistics.mean(ordered):.0f}")
print(f"  分位：p5 {pct(0.05)}  p25 {pct(0.25)}  p50 {pct(0.50)}  p75 {pct(0.75)}  p95 {pct(0.95)}")
print(f"  范围：{ordered[0]} ~ {ordered[-1]}")
print(f"每决策候选数：中位 {statistics.median(all_plans):.0f}")
print(f"各 plan rollout 之和与 total_rollouts 不一致的决策数：{mismatch}（0 = 校验通过）")
print(f"每个候选平均分到 {statistics.mean([t / p for t, p in zip(all_totals, all_plans)]):.1f} 次")
