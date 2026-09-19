"""实验编排 — 单次实验运行 + 多场景批量实验"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

try:
    from .battle_executor import BattleExecutor
    from .bisection_coordinator import BisectionCoordinator, SimplePayoff
    from .individual import (
        generate_individuals,
        get_ground_truth_ranking,
        Individual,
    )
    from .metrics import compute_all_metrics, format_metrics
    from .round_robin import run_round_robin
    from .task_queue import TaskQueue
except ImportError:
    from battle_executor import BattleExecutor  # type: ignore
    from bisection_coordinator import BisectionCoordinator, SimplePayoff  # type: ignore
    from individual import (  # type: ignore
        generate_individuals,
        get_ground_truth_ranking,
        Individual,
    )
    from metrics import compute_all_metrics, format_metrics  # type: ignore
    from round_robin import run_round_robin  # type: ignore
    from task_queue import TaskQueue  # type: ignore


@dataclass
class ExperimentResult:
    """单次实验的完整结果"""
    scenario: str                           # 场景名
    config: dict                            # 配置快照
    n_population: int                       # 个体数
    ground_truth: List[str]                 # Ground Truth 排名
    bisection_ranking: List[str]            # 二分种子排名
    round_robin_ranking: Optional[List[str]]# 全量 RR 排名
    bisection_stats: dict                   # 二分种子统计
    total_battles_bisection: int            # 二分种子总对战局数
    total_battles_rr: Optional[int]         # RR 总对战局数
    bisection_elapsed_s: float              # 二分种子耗时
    rr_elapsed_s: float                     # RR 耗时
    metrics: dict                           # 评估指标
    seed_ids: List[str]                     # 选出的二分种子 ID
    fallback_triggered: bool                # 是否触发 fallback
    individuals_pq: List[Tuple[str, float, float]]  # (id, p, q) 列表


def run_single_experiment(
    n_population: int = 80,
    p_dist: str = "uniform",
    q_mode: str = "constant",
    q_value: float = 8.33,
    bisection_config: Optional[dict] = None,
    run_full_rr: bool = True,
    rr_n_battles: int = 4,
    seed: int = 42,
    scenario_name: str = "default",
) -> ExperimentResult:
    """运行单次实验

    流程:
      1. 生成模拟个体
      2. Ground Truth 排名 (p 降序)
      3. 运行二分种子选拔
      4. (可选) 运行全量 RoundRobin
      5. 计算指标
    """
    t0 = time.time()

    # ── 默认 bisection 配置 ──
    if bisection_config is None:
        bisection_config = {
            "P": 5,
            "M": 20,
            "N": 10,
            "max_seeds": 3,
            "n_battle": 2,
            "max_consecutive_fails": 3,
            "max_workers": 25,
        }

    logger.info(f"{'='*60}")
    logger.info(f"实验: {scenario_name}")
    logger.info(f"N={n_population}, p_dist={p_dist}, q_mode={q_mode}, q_value={q_value}")
    logger.info(f"Bisection config: {bisection_config}")
    logger.info(f"{'='*60}")

    # ── Step 1: 生成个体 ──
    individuals = generate_individuals(
        n=n_population,
        p_dist=p_dist,
        q_mode=q_mode,
        q_value=q_value,
        seed=seed,
    )
    ind_map = {ind.id: ind for ind in individuals}
    logger.info(f"生成 {n_population} 个个体: p ∈ [{min(i.p for i in individuals):.1f}, "
                 f"{max(i.p for i in individuals):.1f}], "
                 f"q ∈ [{min(i.q for i in individuals):.1f}, {max(i.q for i in individuals):.1f}]")

    # ── Step 2: Ground Truth ──
    ground_truth = get_ground_truth_ranking(individuals)
    logger.info(f"Ground Truth Top5: {[f'{iid}(p={ind_map[iid].p:.1f})' for iid in ground_truth[:5]]}")

    # ── Step 3: 二分种子选拔 ──
    logger.info("── 二分种子选拔 ──")
    queue = TaskQueue()
    payoff = SimplePayoff()
    coordinator = BisectionCoordinator(
        config=bisection_config,
        payoff=payoff,
        queue=queue,
    )
    executor = BattleExecutor(
        task_queue=queue,
        individuals=ind_map,
        callback=coordinator.on_battle_complete,
        max_workers=bisection_config.get("max_workers", 25),
    )

    t_bs_start = time.time()
    coordinator.start([ind.id for ind in individuals])
    executor.run()
    t_bs_end = time.time()
    bisection_elapsed = t_bs_end - t_bs_start

    bisection_ranking = coordinator.get_ranking_by_mu()
    seed_ids = list(coordinator.seeds)
    battled_ids = coordinator._get_battled_player_ids()

    bs_stats = coordinator.stats
    total_battles_bs = executor.battles_executed

    logger.info(f"二分种子耗时: {bisection_elapsed:.2f}s")
    logger.info(f"实际对战局数: {total_battles_bs}")
    logger.info(f"已对战个体数 m: {len(battled_ids)} / {len(individuals)}")
    logger.info(f"Fallback: {coordinator.is_fallback}")
    logger.info(f"种子: {seed_ids}")
    logger.info(f"统计: launched={bs_stats['candidates_launched']}, "
                 f"promoted={bs_stats['candidates_promoted']}, "
                 f"failed={bs_stats['candidates_failed']}")

    # ── Step 4: 全量 RoundRobin (可选) ──
    rr_ranking = None
    total_battles_rr = None
    rr_elapsed = 0.0

    if run_full_rr and n_population <= 80:
        logger.info("── 全量 RoundRobin ──")
        rr_payoff = SimplePayoff()
        t_rr_start = time.time()
        rr_payoff, rr_ranking, total_battles_rr = run_round_robin(
            individuals=ind_map,
            participant_ids=[ind.id for ind in individuals],
            n_battles=rr_n_battles,
            payoff=rr_payoff,
            seed=seed + 1000,
        )
        t_rr_end = time.time()
        rr_elapsed = t_rr_end - t_rr_start
        logger.info(f"RR 耗时: {rr_elapsed:.2f}s, 对战局数: {total_battles_rr}")

    elif run_full_rr and n_population > 80:
        logger.warning(f"N={n_population} > 80, 跳过全量 RR (计算量过大)")

    # ── Step 5: 计算指标 ──
    logger.info("── 评估指标 ──")
    metrics = compute_all_metrics(
        ground_truth=ground_truth,
        bisection_ranking=bisection_ranking,
        round_robin_ranking=rr_ranking,
        battled_ids=battled_ids,
    )
    logger.info(f"\n{format_metrics(metrics)}")

    # ── 构建结果 ──
    total_elapsed = time.time() - t0
    logger.info(f"总耗时: {total_elapsed:.2f}s")

    return ExperimentResult(
        scenario=scenario_name,
        config={
            "n_population": n_population,
            "p_dist": p_dist,
            "q_mode": q_mode,
            "q_value": q_value,
            "bisection_config": bisection_config,
            "seed": seed,
        },
        n_population=n_population,
        ground_truth=ground_truth,
        bisection_ranking=bisection_ranking,
        round_robin_ranking=rr_ranking,
        bisection_stats=bs_stats,
        total_battles_bisection=total_battles_bs,
        total_battles_rr=total_battles_rr,
        bisection_elapsed_s=bisection_elapsed,
        rr_elapsed_s=rr_elapsed,
        metrics=metrics,
        seed_ids=seed_ids,
        fallback_triggered=coordinator.is_fallback,
        individuals_pq=[(ind.id, ind.p, ind.q) for ind in individuals],
    )


def run_batch_experiments(
    scenarios: List[dict],
    output_dir: str,
    run_full_rr: bool = True,
) -> List[ExperimentResult]:
    """批量运行多场景实验

    Args:
        scenarios: 场景配置列表，每项包含 n_population, p_dist, q_mode, q_value 等
        output_dir: 结果保存目录
        run_full_rr: 是否运行全量 RR

    Returns:
        所有实验结果列表
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[ExperimentResult] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, sc in enumerate(scenarios):
        scenario_name = sc.get("name", f"scenario_{i:02d}")
        logger.info(f"\n{'#'*60}")
        logger.info(f"场景 [{i+1}/{len(scenarios)}]: {scenario_name}")
        logger.info(f"{'#'*60}")

        result = run_single_experiment(
            n_population=sc.get("n_population", 80),
            p_dist=sc.get("p_dist", "uniform"),
            q_mode=sc.get("q_mode", "constant"),
            q_value=sc.get("q_value", 8.33),
            bisection_config=sc.get("bisection_config"),
            run_full_rr=run_full_rr,
            rr_n_battles=sc.get("rr_n_battles", 4),
            seed=sc.get("seed", 42 + i),
            scenario_name=scenario_name,
        )
        results.append(result)

        # 每个场景单独保存
        scene_file = os.path.join(output_dir, f"{timestamp}_{scenario_name}.json")
        save_result(result, scene_file)
        logger.info(f"场景结果已保存: {scene_file}")

    # 汇总报告
    summary_file = os.path.join(output_dir, f"{timestamp}_summary.json")
    save_summary(results, summary_file)
    logger.info(f"汇总报告已保存: {summary_file}")

    # 也保存一份 latest 副本
    latest_file = os.path.join(output_dir, "latest_summary.json")
    save_summary(results, latest_file)

    return results


def save_result(result: ExperimentResult, filepath: str) -> None:
    """保存单次实验结果到 JSON"""
    export = {
        "scenario": result.scenario,
        "config": result.config,
        "n_population": result.n_population,
        "metrics": result.metrics,
        "total_battles_bisection": result.total_battles_bisection,
        "total_battles_rr": result.total_battles_rr,
        "bisection_elapsed_s": round(result.bisection_elapsed_s, 3),
        "rr_elapsed_s": round(result.rr_elapsed_s, 3),
        "bisection_stats": result.bisection_stats,
        "seed_ids": result.seed_ids,
        "fallback_triggered": result.fallback_triggered,
        "ground_truth_top10": result.ground_truth[:10],
        "bisection_top10": result.bisection_ranking[:10],
        "round_robin_top10": result.round_robin_ranking[:10] if result.round_robin_ranking else None,
    }
    with open(filepath, "w") as f:
        json.dump(export, f, indent=2, ensure_ascii=False)


def save_summary(results: List[ExperimentResult], filepath: str) -> None:
    """保存批量实验的汇总报告到 JSON"""
    rows = []
    for r in results:
        rows.append({
            "scenario": r.scenario,
            "n_population": r.n_population,
            "p_dist": r.config.get("p_dist"),
            "q_mode": r.config.get("q_mode"),
            "q_value": r.config.get("q_value"),
            "metrics": r.metrics,
            "total_battles_bisection": r.total_battles_bisection,
            "total_battles_rr": r.total_battles_rr,
            "bisection_elapsed_s": round(r.bisection_elapsed_s, 3),
            "rr_elapsed_s": round(r.rr_elapsed_s, 3),
            "n_seeds_found": len(r.seed_ids),
            "fallback_triggered": r.fallback_triggered,
            "bisection_stats": r.bisection_stats,
        })

    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(results),
        "results": rows,
    }
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
