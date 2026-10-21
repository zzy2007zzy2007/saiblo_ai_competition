"""统计一个日志里的决策级信息，用来核对"亚军决策机制"的理解。

关注：
- 每决策总 rollout 数（时间预算内实际跑了多少）
- 每决策候选数
- 每个 arm 的 rollout 数分布（探针 5 → 公共批次 → UCB 追加，上限 125）
- 后置覆盖（emp / evasion / ant_upgrade）触发频率
- 是否真的会选中"带闪电"的计划（这些 arm 只有 2~4 次采样）
"""
import collections
import json
import statistics
import sys

path = sys.argv[1]
decisions = 0
totals = []
plans = []
actions = []
emp = evasion = ant_upgrade = 0
per_arm = collections.Counter()
chosen = collections.Counter()
lightning_chosen = 0
rollout_by_rank1 = []

for line in open(path, encoding="utf-8", errors="replace"):
    if '"kind":"decision"' in line:
        record = json.loads(line)
        decisions += 1
        totals.append(record["total_rollouts"])
        plans.append(record["plans_total"])
        actions.append(record["action_count"])
        emp += int(record["v4_emp_used"])
        evasion += int(record["v4_evasion_used"])
        ant_upgrade += int(record["v4_ant_upgrade_used"])
    elif "base_heuristic" in line:
        record = json.loads(line)
        per_arm[record["rollouts"]] += 1
        if record.get("rank") == 1:
            key = record["key"]
            chosen[key.split("|")[0]] += 1
            rollout_by_rank1.append(record["rollouts"])
            if any(segment.startswith("21:") for segment in key.split("|")):
                lightning_chosen += 1

print(f"决策数 = {decisions}")
print(f"每决策 total_rollouts: 中位 {statistics.median(totals):.0f} "
      f"均值 {statistics.mean(totals):.0f} 范围 [{min(totals)}, {max(totals)}]")
print(f"每决策候选数: 中位 {statistics.median(plans):.0f} 范围 [{min(plans)}, {max(plans)}]")
print(f"每决策实际下发操作数: 中位 {statistics.median(actions):.0f} "
      f"分布 {collections.Counter(actions).most_common(6)}")
print("\n每个 arm 的 rollout 数分布（前 10）:")
for count, freq in per_arm.most_common(10):
    print(f"   {count:>4} 次 : {freq}")
print(f"\n后置覆盖: emp {emp} 次 / evasion {evasion} 次 / ant_upgrade {ant_upgrade} 次")
print(f"选中计划里含闪电(op 21) 的决策数 = {lightning_chosen}")
print(f"rank1 的 rollout 数: 中位 {statistics.median(rollout_by_rank1):.0f} "
      f"范围 [{min(rollout_by_rank1)}, {max(rollout_by_rank1)}]")
print(f"rank1 的 op 类型分布（前 8）: {chosen.most_common(8)}")
