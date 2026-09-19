"""original_rule 方案验证实验入口（v2 — 精简指标）

矩阵：3 (p_dist) × 3 (q_mode) × 5 (seed) = 45 组实验
固定参数：n=80, M=16, max_seeds=3, n_battle=1, max_candidate_fails=12

每组实验同时执行：
  1) 二分种子方案 — original_rule 协调器
  2) 全量 RoundRobin（n_battle=1）— 作为对照基线

精简后的核心指标：
  - 计算规模：总对战 / 一类 / 二类 / 三类 / 有效 / 累计候选个体数 / 节省率 / Fallback
  - 覆盖率：Top16 重叠率 vs GT、Recall@8、Recall@16
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
from loguru import logger

# 复用 bisection_experiment 的基础组件
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BISECTION_DIR = os.path.join(_THIS_DIR, "..", "bisection_experiment")
if _BISECTION_DIR not in sys.path:
    sys.path.insert(0, _BISECTION_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from individual import (  # type: ignore
    generate_individuals,
    get_ground_truth_ranking,
)
from task_queue import TaskQueue  # type: ignore
from battle_executor import BattleExecutor  # type: ignore
from round_robin import run_round_robin  # type: ignore

from coordinator import OriginalRuleCoordinator, SimplePayoff  # type: ignore


# ── 指标 ──

def topk_overlap(ranking_a: List[str], ranking_b: List[str], k: int) -> float:
    """TopK 重叠率：|a[:k] ∩ b[:k]| / k"""
    return len(set(ranking_a[:k]) & set(ranking_b[:k])) / k


def recall_within_top(
    algo_ranking: List[str],
    gt_ranking: List[str],
    pool_k: int,
    target_k: int,
) -> float:
    """Recall@target_k within Top-pool_k = |algo[:pool_k] ∩ gt[:target_k]| / target_k"""
    return len(set(algo_ranking[:pool_k]) & set(gt_ranking[:target_k])) / target_k


# ── 数据结构 ──

@dataclass
class ScenarioResult:
    name: str
    p_dist: str
    q_mode: str
    seed: int
    # 计算规模
    battles_total: int
    battles_class1: int     # 一类：提交时一方是候选（无种子参与）
    battles_class2: int     # 二类：提交时一方是正式种子
    battles_class3: int     # 三类：执行完毕时双方都不是正式种子
    battles_effective: int  # 有效 = 总数 − 三类
    unique_candidates: int  # 累计候选个体数（去重）
    rr_battles: int
    saving_rate: float
    fallback_triggered: bool
    seed_ids: List[str]
    # 覆盖率
    top16_overlap_vs_gt: float
    recall8_in_top16: float
    recall16_in_top16: float


def run_one_scenario(
    name: str,
    p_dist: str,
    q_mode: str,
    q_value: float,
    seed: int,
    config: dict,
) -> ScenarioResult:
    individuals = generate_individuals(
        n=config["n_population"],
        p_dist=p_dist,
        q_mode=q_mode,
        q_value=q_value,
        seed=seed,
    )
    ind_map = {ind.id: ind for ind in individuals}
    gt_ranking = get_ground_truth_ranking(individuals)

    queue = TaskQueue()
    payoff = SimplePayoff()
    coord = OriginalRuleCoordinator(config=config, payoff=payoff, queue=queue)
    executor = BattleExecutor(
        task_queue=queue,
        individuals=ind_map,
        callback=coord.on_battle_complete,
        max_workers=config.get("max_workers", 25),
    )

    coord.start([ind.id for ind in individuals])
    executor.run()
    coord.finalize_battle_classification()

    bisection_ranking = coord.get_ranking_by_mu()

    rr_payoff = SimplePayoff()
    _, _, rr_battles = run_round_robin(
        individuals=ind_map,
        participant_ids=[ind.id for ind in individuals],
        n_battles=config.get("n_battle", 1),
        payoff=rr_payoff,
        seed=seed + 10000,
    )

    saving_rate = 1.0 - executor.battles_executed / max(rr_battles, 1)

    return ScenarioResult(
        name=name,
        p_dist=p_dist,
        q_mode=q_mode,
        seed=seed,
        battles_total=coord.stats["battles_total"],
        battles_class1=coord.stats["battles_class1"],
        battles_class2=coord.stats["battles_class2"],
        battles_class3=coord.stats["battles_class3"],
        battles_effective=coord.stats["battles_effective"],
        unique_candidates=coord.unique_candidates_count,
        rr_battles=rr_battles,
        saving_rate=saving_rate,
        fallback_triggered=coord.is_fallback,
        seed_ids=list(coord.seeds),
        top16_overlap_vs_gt=topk_overlap(bisection_ranking, gt_ranking, 16),
        recall8_in_top16=recall_within_top(bisection_ranking, gt_ranking, 16, 8),
        recall16_in_top16=recall_within_top(bisection_ranking, gt_ranking, 16, 16),
    )


# ── 矩阵 ──

P_DISTS = ["uniform", "normal", "bimodal"]
Q_MODES = [("constant", 8.33), ("proportional", 8.33), ("random", 8.33)]
SEEDS = [42, 142, 242, 342, 442]


# ── 报告 ──

def setup_logging(output_dir: str, level: str = "INFO") -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(log_file, level="DEBUG", rotation="50 MB", encoding="utf-8")
    return log_file


def print_per_scenario_battles_table(results: List[ScenarioResult]) -> None:
    """45 组对局数量明细表"""
    sep = "─" * 110
    logger.info(sep)
    logger.info("【表 1】45 组实验 — 对局数量明细（按场景）")
    logger.info(sep)
    header = (f"{'#':>2}  {'场景':<32} "
              f"{'总数':>5} {'一类':>5} {'二类':>5} {'三类':>5} "
              f"{'有效':>5} {'候选':>4} {'RR':>5} {'节省%':>7} {'Fb':>3}")
    logger.info(header)
    logger.info(sep)
    for i, r in enumerate(results, 1):
        logger.info(
            f"{i:>2}  {r.name:<32} "
            f"{r.battles_total:>5d} {r.battles_class1:>5d} {r.battles_class2:>5d} "
            f"{r.battles_class3:>5d} {r.battles_effective:>5d} "
            f"{r.unique_candidates:>4d} {r.rr_battles:>5d} "
            f"{r.saving_rate*100:>6.2f}% "
            f"{'Y' if r.fallback_triggered else 'N':>3}"
        )
    logger.info(sep)


def print_per_scenario_coverage_table(results: List[ScenarioResult]) -> None:
    """45 组覆盖率明细表"""
    sep = "─" * 90
    logger.info(sep)
    logger.info("【表 2】45 组实验 — 覆盖率明细（按场景）")
    logger.info(sep)
    header = (f"{'#':>2}  {'场景':<32} "
              f"{'Top16重叠':>10} {'Recall8':>9} {'Recall16':>10}")
    logger.info(header)
    logger.info(sep)
    for i, r in enumerate(results, 1):
        logger.info(
            f"{i:>2}  {r.name:<32} "
            f"{r.top16_overlap_vs_gt:>10.4f} "
            f"{r.recall8_in_top16:>9.4f} "
            f"{r.recall16_in_top16:>10.4f}"
        )
    logger.info(sep)


def mean_std(vals):
    arr = [v for v in vals if v is not None]
    if not arr:
        return 0.0, 0.0
    return float(np.mean(arr)), float(np.std(arr))


def print_aggregate(results: List[ScenarioResult]) -> dict:
    """整体 + 分组汇总"""
    sep = "═" * 110

    logger.info(sep)
    logger.info("【表 3】整体汇总（45 组平均 ± 标准差）")
    logger.info(sep)
    fields = [
        ("总对战数", [r.battles_total for r in results]),
        ("一类对战数（提交时一方是候选）", [r.battles_class1 for r in results]),
        ("二类对战数（提交时一方是正式种子）", [r.battles_class2 for r in results]),
        ("三类对战数（结束时双方都非种子）", [r.battles_class3 for r in results]),
        ("有效对战数（= 总数 − 三类）", [r.battles_effective for r in results]),
        ("累计候选个体数（去重）", [r.unique_candidates for r in results]),
        ("RR 对战数", [r.rr_battles for r in results]),
        ("节省率 (%)", [r.saving_rate * 100 for r in results]),
        ("Top16 重叠率 vs GT", [r.top16_overlap_vs_gt for r in results]),
        ("Recall@8 within Top16", [r.recall8_in_top16 for r in results]),
        ("Recall@16 within Top16", [r.recall16_in_top16 for r in results]),
    ]
    overall = {}
    for name, vals in fields:
        mean, std = mean_std(vals)
        overall[name] = {"mean": mean, "std": std}
        logger.info(f"  {name:<40} : {mean:>10.4f} ± {std:>8.4f}")
    fb_rate = sum(1 for r in results if r.fallback_triggered) / max(len(results), 1)
    overall["Fallback 触发率"] = fb_rate
    logger.info(f"  {'Fallback 触发率':<40} : {fb_rate*100:>10.2f}%")
    logger.info(sep)

    # 按 p 分布分组
    logger.info("【表 4】按 p 分布分组")
    logger.info(sep)
    header = (f"  {'group':<14} {'n':>2} "
              f"{'总数':>7} {'一类':>6} {'二类':>6} {'三类':>6} {'有效':>7} "
              f"{'候选':>5} {'节省%':>7} {'Top16':>7} {'R@8':>6} {'R@16':>7}")
    logger.info(header)
    by_group: Dict[str, List[ScenarioResult]] = {}
    for r in results:
        by_group.setdefault(r.p_dist, []).append(r)
    for k in sorted(by_group):
        g = by_group[k]
        logger.info(
            f"  p={k:<12} {len(g):>2d} "
            f"{np.mean([r.battles_total for r in g]):>7.1f} "
            f"{np.mean([r.battles_class1 for r in g]):>6.1f} "
            f"{np.mean([r.battles_class2 for r in g]):>6.1f} "
            f"{np.mean([r.battles_class3 for r in g]):>6.1f} "
            f"{np.mean([r.battles_effective for r in g]):>7.1f} "
            f"{np.mean([r.unique_candidates for r in g]):>5.1f} "
            f"{np.mean([r.saving_rate*100 for r in g]):>6.2f}% "
            f"{np.mean([r.top16_overlap_vs_gt for r in g]):>7.4f} "
            f"{np.mean([r.recall8_in_top16 for r in g]):>6.4f} "
            f"{np.mean([r.recall16_in_top16 for r in g]):>7.4f}"
        )
    logger.info(sep)

    # 按 q 模式分组
    logger.info("【表 5】按 q 模式分组")
    logger.info(sep)
    logger.info(header)
    by_group = {}
    for r in results:
        by_group.setdefault(r.q_mode, []).append(r)
    for k in sorted(by_group):
        g = by_group[k]
        logger.info(
            f"  q={k:<12} {len(g):>2d} "
            f"{np.mean([r.battles_total for r in g]):>7.1f} "
            f"{np.mean([r.battles_class1 for r in g]):>6.1f} "
            f"{np.mean([r.battles_class2 for r in g]):>6.1f} "
            f"{np.mean([r.battles_class3 for r in g]):>6.1f} "
            f"{np.mean([r.battles_effective for r in g]):>7.1f} "
            f"{np.mean([r.unique_candidates for r in g]):>5.1f} "
            f"{np.mean([r.saving_rate*100 for r in g]):>6.2f}% "
            f"{np.mean([r.top16_overlap_vs_gt for r in g]):>7.4f} "
            f"{np.mean([r.recall8_in_top16 for r in g]):>6.4f} "
            f"{np.mean([r.recall16_in_top16 for r in g]):>7.4f}"
        )
    logger.info(sep)

    return overall


# ── 主入口 ──

def main():
    parser = argparse.ArgumentParser(description="original_rule 方案验证实验 (v2)")
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--seeds", type=str, default="42,142,242,342,442")
    parser.add_argument("--p-dists", type=str, default="uniform,normal,bimodal")
    parser.add_argument("--q-modes", type=str, default="constant,proportional,random")
    parser.add_argument("--output-dir", type=str,
                        default="ppo_v10/outputs/original_rule_experiment_v2")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    log_file = setup_logging(output_dir)
    logger.info(f"日志: {log_file}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    p_dists = [s.strip() for s in args.p_dists.split(",")]
    q_modes_raw = [s.strip() for s in args.q_modes.split(",")]
    q_modes = [(q, 8.33) for q in q_modes_raw]

    config = {
        "n_population": args.n,
        "M": 32,
        "max_seeds": 3,
        "n_battle": 2,
        "max_candidate_fails": 12,
        "max_workers": 25,
    }

    scenarios = []
    for p in p_dists:
        for qm, qv in q_modes:
            for s in seeds:
                scenarios.append({
                    "name": f"N{args.n}_{p}_{qm}_s{s}",
                    "p_dist": p, "q_mode": qm, "q_value": qv, "seed": s,
                })

    logger.info(f"场景总数: {len(scenarios)}, 配置: {config}")

    results: List[ScenarioResult] = []
    t0 = time.time()
    for i, sc in enumerate(scenarios):
        logger.info(f"[{i+1}/{len(scenarios)}] {sc['name']}")
        try:
            r = run_one_scenario(
                name=sc["name"],
                p_dist=sc["p_dist"],
                q_mode=sc["q_mode"],
                q_value=sc["q_value"],
                seed=sc["seed"],
                config=config,
            )
            results.append(r)
            logger.info(
                f"   total={r.battles_total} c1={r.battles_class1} c2={r.battles_class2} "
                f"c3={r.battles_class3} eff={r.battles_effective} cand={r.unique_candidates} "
                f"top16={r.top16_overlap_vs_gt:.3f} R@8={r.recall8_in_top16:.3f}"
            )
        except Exception as e:
            logger.exception(f"场景 {sc['name']} 失败: {e}")

    total_elapsed = time.time() - t0

    # 输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_file = os.path.join(output_dir, f"{timestamp}_results.json")
    with open(detail_file, "w") as f:
        json.dump(
            [asdict(r) for r in results],
            f, indent=2, ensure_ascii=False,
            default=lambda x: float(x) if isinstance(x, np.floating) else str(x),
        )
    logger.info(f"详细结果已保存: {detail_file}")

    print_per_scenario_battles_table(results)
    print_per_scenario_coverage_table(results)
    overall = print_aggregate(results)

    summary = {
        "timestamp": timestamp,
        "config": config,
        "n_scenarios": len(results),
        "total_elapsed_s": round(total_elapsed, 2),
        "overall": overall,
        "per_scenario": [asdict(r) for r in results],
    }
    summary_file = os.path.join(output_dir, f"{timestamp}_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False,
                  default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    logger.info(f"汇总报告: {summary_file}")
    logger.info(f"实验完成 | 总耗时 {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
