"""统计"亚军一次决策有多少候选动作真的被 rollout"。

对每个决策统计：
- plans_total：候选计划数（generate_root_plans 产出的）
- evaluated：日志里出现的 plan 行数（= 被采样到、进入 evaluated 的臂数）
- total_rollouts：这次决策一共跑了多少次 rollout
- 每臂采样数的分布（各分位）
- rollout 预算的集中度：top-10 臂吃掉了多少比例

用法：
    python probe_candidate_counts.py match_results/ai0_runnerup_seed732.stderr.log [...]
"""
from __future__ import annotations

import collections
import json
import statistics
import sys


def summarize(path: str) -> list[dict]:
    """⚠️ 日志里 plan 行在 decision 行**之前**打印（decision.hpp:70-141），
    所以不能一看到 decision 行就开始收集臂——那样会串到下一个决策。
    正确做法：先把臂攒在 pending 里，等下一个 decision 行到来时再结账。
    """
    decisions: list[dict] = []
    pending: list[dict] = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if '"kind":"decision"' in line:
            record = json.loads(line)
            decisions.append({
                "plans_total": record["plans_total"],
                "total_rollouts": record["total_rollouts"],
                "arms": pending,
            })
            pending = []
        elif "base_heuristic" in line:
            record = json.loads(line)
            pending.append({
                "rollouts": record["rollouts"],
                "rank": record.get("rank"),
                "key": record["key"],
                "has_lightning": any(s.startswith("21:") for s in record["key"].split("|")),
            })
    return decisions


def main() -> int:
    all_plans: list[int] = []
    all_evaluated: list[int] = []
    all_totals: list[int] = []
    all_arm_counts: list[int] = []
    arms_ge100: list[int] = []
    arms_ge20: list[int] = []
    dropped = 0
    top10_share: list[float] = []
    lightning_counts: list[int] = []
    bucket = collections.Counter()
    per_arm_rollouts: collections.Counter = collections.Counter()

    for path in sys.argv[1:]:
        decisions = summarize(path)
        print(f"[{path.split(chr(92))[-1]}] 决策数 {len(decisions)}")
        for decision in decisions:
            arms = decision["arms"]
            if not arms:
                continue
            plans_total = decision["plans_total"]
            all_plans.append(plans_total)
            all_evaluated.append(len(arms))
            all_totals.append(decision["total_rollouts"])
            if plans_total != len(arms):
                dropped += 1
            counts = sorted((arm["rollouts"] for arm in arms), reverse=True)
            all_arm_counts.extend(counts)
            arms_ge100.append(sum(1 for count in counts if count >= 100))
            arms_ge20.append(sum(1 for count in counts if count >= 20))
            lightning_counts.append(sum(1 for arm in arms if arm["has_lightning"]))
            for count in counts:
                per_arm_rollouts[count] += 1
            total = sum(counts) or 1
            top10_share.append(sum(counts[:10]) / total)
            for count in counts:
                if count <= 5:
                    bucket["1-5"] += 1
                elif count <= 20:
                    bucket["6-20"] += 1
                elif count <= 99:
                    bucket["21-99"] += 1
                elif count <= 199:
                    bucket["100-199"] += 1
                elif count <= 499:
                    bucket["200-499"] += 1
                else:
                    bucket[">=500"] += 1

    def line(name: str, values: list[int]) -> str:
        ordered = sorted(values)
        return (f"  {name:<22} 中位 {statistics.median(ordered):>7.0f}  "
                f"p10 {ordered[int(0.1 * len(ordered))]:>7}  p90 {ordered[int(0.9 * len(ordered))]:>7}  "
                f"范围 [{ordered[0]}, {ordered[-1]}]")

    print(f"\n=== {len(all_plans)} 个决策 ===")
    print(line("候选计划数", all_plans))
    print(line("实际进入 evaluated", all_evaluated))
    print(line("拿到 >=20 次的臂数", arms_ge20))
    print(line("拿到 >=100 次的臂数", arms_ge100))
    print(line("每决策 rollout 总数", all_totals))
    print(line("闪电臂数", lightning_counts))
    print(f"\n候选数 vs 被评估数不一致的决策: {dropped}（说明有臂因非法/采样失败被丢弃）")
    print(f"top-10 臂占用的 rollout 比例: 中位 {statistics.median(top10_share) * 100:.1f}%")

    print("\n每个候选臂拿到的 rollout 数分布:")
    for name in ("1-5", "6-20", "21-99", "100-199", "200-499", ">=500"):
        count = bucket[name]
        print(f"  {name:<9} {count:>8} 个臂  {100.0 * count / max(sum(bucket.values()), 1):5.1f}%")

    print("\n最常见的单臂 rollout 数（前 8）:")
    for value, freq in per_arm_rollouts.most_common(8):
        print(f"  {value:>5} 次 : {freq:>8} 个臂")

    print("\n=== 换算：一次决策大约 roll out 多少「不同的候选动作」 ===")
    evaluated_median = statistics.median(all_evaluated)
    print(f"  {evaluated_median:.0f} 个候选各拿到 ≥1 次 rollout（其中"
          f"{bucket['100-199'] + bucket['200-499'] + bucket['>=500']} 个臂次拿到 ≥100 次）")
    print(f"  实际总 rollout 中位 {statistics.median(all_totals):.0f} 次")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
