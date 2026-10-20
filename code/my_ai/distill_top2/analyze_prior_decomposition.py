"""从亚军的 plans 日志里拆解「一个决策到底由先验还是由搜索决定」。

日志里每个决策有 ~70-200 条 `{"kind":"plan",...}` 行，关键字段：
    heuristic           ← 先验（纯函数、手调常数表）
    mean_terminal_score ← 搜索（~125 次 rollout 的终止评估均值）
    score               ← root_score = mean_terminal_score + heuristic

要回答：
 1. 先验的取值有多稀疏（是不是几乎只有 {0, 80}），各分项非零比例
 2. **只看先验的 argmax vs 只看 score 的 argmax 一致率** ⇒ 先验对决策的实际贡献
 3. top1 / top2 的分差分布 ⇒ 先验的 80 分是否常能翻转结论
 4. 先验翻盘的例子（对定位先验到底靠什么规则生效很有用）

用法：
    python analyze_prior_decomposition.py match_results/ai0_runnerup_seed732.stderr.log
    python analyze_prior_decomposition.py <log> --max-rounds 100
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path


def iter_decisions(path: Path):
    """按 (round, player) 聚合日志里的 plan 行（日志是按决策连续写的）。"""
    current_key = None
    current_plans: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "base_heuristic" not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (record.get("round"), record.get("player"))
            if key != current_key:
                if current_plans:
                    yield current_key, current_plans
                current_key = key
                current_plans = []
            current_plans.append(record)
    if current_plans:
        yield current_key, current_plans


def top_by(plans: list[dict], field: str) -> dict:
    return max(plans, key=lambda plan: plan[field])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--examples", type=int, default=6)
    args = parser.parse_args()

    component_fields = ["base_heuristic", "lure_heuristic", "lightning_heuristic",
                        "operation_penalty", "followup_penalty", "lightning_static_bonus"]
    nonzero_count: collections.Counter = collections.Counter()
    total_plans = 0
    heuristic_values: collections.Counter = collections.Counter()
    n_decisions = 0
    agree_prior = 0
    agree_search = 0
    prior_flips: list[dict] = []
    score_gaps: list[float] = []
    heur_gaps: list[float] = []
    identity_ok = 0
    identity_total = 0
    nplan_hist: list[int] = []
    chosen_keys: collections.Counter = collections.Counter()
    rank1_keys: collections.Counter = collections.Counter()
    rank1_matches_max = 0
    rank1_total = 0

    for path_str in args.logs:
        path = Path(path_str)
        print(f"[读] {path.name}")
        for key, plans in iter_decisions(path):
            round_index = key[0]
            if args.max_rounds and round_index is not None and round_index >= args.max_rounds:
                continue
            if not plans:
                continue
            n_decisions += 1
            total_plans += len(plans)
            nplan_hist.append(len(plans))
            for plan in plans:
                heuristic_values[round(float(plan["heuristic"]), 4)] += 1
                for field in component_fields:
                    if abs(float(plan.get(field, 0.0))) > 1e-9:
                        nonzero_count[field] += 1
                identity_total += 1
                # score 的完整构成：mean_terminal_score + heuristic + 闪电加成 − 反应式惩罚
                reconstructed = (plan["mean_terminal_score"] + plan["heuristic"]
                                 + float(plan.get("mean_lightning_bonus_score", 0.0))
                                 - float(plan.get("mean_reactive_operation_penalty", 0.0)))
                if abs(plan["score"] - reconstructed) < 1e-6:
                    identity_ok += 1

            by_score = top_by(plans, "score")
            by_heur = top_by(plans, "heuristic")
            by_search = top_by(plans, "mean_terminal_score")
            chosen_keys[by_score["key"].split("|")[0]] += 1
            rank1 = [plan for plan in plans if plan.get("rank") == 1]
            if rank1:
                rank1_keys[rank1[0]["key"].split("|")[0]] += 1
                if rank1[0]["key"] == by_score["key"]:
                    rank1_matches_max += 1
                rank1_total += 1
            if by_heur["key"] == by_score["key"]:
                agree_prior += 1
            if by_search["key"] == by_score["key"]:
                agree_search += 1
            if by_search["key"] != by_score["key"] and by_heur["key"] == by_score["key"]:
                prior_flips.append({
                    "round": key[0], "player": key[1],
                    "chosen": by_score["key"],
                    "search_best": by_search["key"],
                    "chosen_h": by_score["heuristic"],
                    "search_best_h": by_search["heuristic"],
                    "score_gap": by_score["score"] - by_search["score"],
                })

            ordered = sorted(plans, key=lambda plan: -plan["score"])
            if len(ordered) >= 2:
                score_gaps.append(ordered[0]["score"] - ordered[1]["score"])
            ordered_h = sorted(plans, key=lambda plan: -plan["heuristic"])
            if len(ordered_h) >= 2:
                heur_gaps.append(ordered_h[0]["heuristic"] - ordered_h[1]["heuristic"])

    if not n_decisions:
        print("没读到决策", flush=True)
        return 2

    def pct(num: int) -> str:
        return f"{100.0 * num / max(n_decisions, 1):.1f}%"

    print(f"\n=== 规模 ===")
    print(f"决策数（round×player）= {n_decisions}   候选计划总数 = {total_plans}")
    print(f"每决策候选数: 均值 {statistics.mean(nplan_hist):.1f} "
          f"中位 {statistics.median(nplan_hist)} 范围 [{min(nplan_hist)}, {max(nplan_hist)}]")

    print(f"\n=== 先验取值分布（前 12）===")
    for value, count in heuristic_values.most_common(12):
        print(f"  {value:>10}: {count:>8}  {100.0 * count / total_plans:5.2f}%")
    print(f"  不同取值个数 = {len(heuristic_values)}")

    print(f"\n=== 分项非零比例 ===")
    for field in component_fields:
        print(f"  {field:<24} {100.0 * nonzero_count[field] / total_plans:6.2f}%")

    print(f"\n=== 决策归因 ===")
    print(f"argmax(score) == argmax(heuristic)      : {agree_prior:>5} / {n_decisions}  {pct(agree_prior)}")
    print(f"argmax(score) == argmax(mean_terminal)  : {agree_search:>5} / {n_decisions}  {pct(agree_search)}")
    print(f"先验翻盘（搜索最佳 ≠ 最终选择）        : {len(prior_flips):>5} / {n_decisions}  {pct(len(prior_flips))}")

    print(f"\n=== 分差 ===")
    if score_gaps:
        print(f"top1-top2 score 差: 均值 {statistics.mean(score_gaps):.1f} "
              f"中位 {statistics.median(score_gaps):.1f} p90 {sorted(score_gaps)[int(0.9 * len(score_gaps))]:.1f}")
    if heur_gaps:
        print(f"top1-top2 heuristic 差: 均值 {statistics.mean(heur_gaps):.1f} "
              f"中位 {statistics.median(heur_gaps):.1f} p90 {sorted(heur_gaps)[int(0.9 * len(heur_gaps))]:.1f}")
    print(f"\n=== 恒等式 score == mean_terminal + heuristic + 闪电 − 反应式 ===")
    print(f"  成立 {identity_ok} / {identity_total}")

    print(f"\n=== 实际选择分布（按 score 最大，前 10；key 截到第一个 op）===")
    for key, count in chosen_keys.most_common(10):
        print(f"  {key:<28} {count:>5}  {100.0 * count / n_decisions:5.1f}%")
    if rank1_total:
        print(f"  [校验] rank==1 与 max(score) 一致 {rank1_matches_max} / {rank1_total}")
        print("  rank==1 的 key 分布（前 6）: " + ", ".join(
            f"{k}:{v}" for k, v in rank1_keys.most_common(6)))

    if prior_flips and args.examples:
        print(f"\n=== 先验翻盘例子（前 {args.examples} 个）===")
        for item in prior_flips[:args.examples]:
            print(f"  r{item['round']:<4} p{item['player']} 选 {item['chosen']}(h={item['chosen_h']}) "
                  f"而搜索最佳 {item['search_best']}(h={item['search_best_h']}) 分差 {item['score_gap']:+.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
