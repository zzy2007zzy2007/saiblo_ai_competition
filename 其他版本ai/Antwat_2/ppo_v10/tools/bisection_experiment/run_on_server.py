#!/usr/bin/env python3
"""服务器端运行器 — 独立运行，不依赖包结构

用法:
    python3 run_on_server.py --preset default
    python3 run_on_server.py --n 80 --p-dist uniform --full-rr
"""

import json
import os
import sys
import time
from datetime import datetime

# 确保当前目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger

# 直接导入（flat structure，无包相对导入）
from individual import generate_individuals, get_ground_truth_ranking
from task_queue import TaskQueue
from bisection_coordinator import BisectionCoordinator, SimplePayoff
from battle_executor import BattleExecutor
from metrics import compute_all_metrics, format_metrics
from round_robin import run_round_robin


def get_preset_scenarios(preset: str) -> list:
    """预设场景列表"""
    base_config = {
        "P": 5, "M": 20, "N": 10, "max_seeds": 3,
        "n_battle": 2, "max_consecutive_fails": 3, "max_workers": 25,
    }

    # P=3, max_consecutive_fails=12 的调优配置
    n80_config = {
        "P": 3, "M": 20, "N": 10, "max_seeds": 3,
        "n_battle": 2, "max_consecutive_fails": 12, "max_workers": 25,
    }

    if preset == "quick":
        return [
            {"name": "quick_N40_u8", "n_population": 40, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": dict(base_config), "seed": 42},
        ]
    elif preset == "default":
        return [
            {"name": "N40_uniform", "n_population": 40, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": dict(base_config), "seed": 42},
            {"name": "N80_uniform", "n_population": 80, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": dict(base_config), "seed": 43},
            {"name": "N80_normal", "n_population": 80, "p_dist": "normal",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": dict(base_config), "seed": 44},
            {"name": "N80_bimodal", "n_population": 80, "p_dist": "bimodal",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": dict(base_config), "seed": 45},
            {"name": "N20_uniform", "n_population": 20, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": dict(base_config), "seed": 46},
        ]
    elif preset == "n80":
        # 仅 N=80，新参数 P=3, max_consecutive_fails=6
        scenarios = []
        for p_dist in ["uniform", "normal", "bimodal"]:
            scenarios.append({
                "name": f"N80_{p_dist}",
                "n_population": 80, "p_dist": p_dist,
                "q_mode": "constant", "q_value": 8.33,
                "bisection_config": dict(n80_config),
                "seed": 42 + len(scenarios),
            })
        for q_mode in ["proportional", "random"]:
            for q_value in [5.0, 12.0]:
                scenarios.append({
                    "name": f"N80_uniform_q{q_mode}_{q_value}",
                    "n_population": 80, "p_dist": "uniform",
                    "q_mode": q_mode, "q_value": q_value,
                    "bisection_config": dict(n80_config),
                    "seed": 100 + len(scenarios),
                })
        return scenarios
    elif preset == "full":
        scenarios = []
        for n in [20, 40, 80]:
            for p_dist in ["uniform", "normal", "bimodal"]:
                scenarios.append({
                    "name": f"N{n}_{p_dist}",
                    "n_population": n, "p_dist": p_dist,
                    "q_mode": "constant", "q_value": 8.33,
                    "bisection_config": dict(base_config),
                    "seed": 42 + len(scenarios),
                })
        for q_mode in ["proportional", "random"]:
            for q_value in [5.0, 12.0]:
                scenarios.append({
                    "name": f"N80_uniform_q{q_mode}_{q_value}",
                    "n_population": 80, "p_dist": "uniform",
                    "q_mode": q_mode, "q_value": q_value,
                    "bisection_config": dict(base_config),
                    "seed": 100 + len(scenarios),
                })
        return scenarios
    else:
        raise ValueError(f"Unknown preset: {preset}")


def run_single(n_population, p_dist, q_mode, q_value, bisection_config, run_full_rr, seed, scenario_name):
    """运行单次实验"""
    logger.info(f"{'='*60}")
    logger.info(f"实验: {scenario_name}")
    logger.info(f"N={n_population}, p_dist={p_dist}, q_mode={q_mode}, q_value={q_value}")
    logger.info(f"{'='*60}")

    # Step 1: 生成个体
    individuals = generate_individuals(
        n=n_population, p_dist=p_dist, q_mode=q_mode, q_value=q_value, seed=seed)
    ind_map = {ind.id: ind for ind in individuals}
    p_vals = [ind.p for ind in individuals]
    q_vals = [ind.q for ind in individuals]
    logger.info(f"生成 {n_population} 个体: p∈[{min(p_vals):.1f}, {max(p_vals):.1f}], "
                 f"q∈[{min(q_vals):.2f}, {max(q_vals):.2f}]")

    # Step 2: Ground Truth
    ground_truth = get_ground_truth_ranking(individuals)

    # Step 3: 二分种子选拔
    logger.info("── 二分种子选拔 ──")
    queue = TaskQueue()
    payoff = SimplePayoff()
    coordinator = BisectionCoordinator(config=bisection_config, payoff=payoff, queue=queue)
    executor = BattleExecutor(
        task_queue=queue, individuals=ind_map,
        callback=coordinator.on_battle_complete,
        max_workers=bisection_config.get("max_workers", 25),
    )

    t0 = time.time()
    coordinator.start([ind.id for ind in individuals])
    executor.run()
    coordinator.finalize_battle_classification()  # 回扫类 3
    bs_elapsed = time.time() - t0

    bisection_ranking = coordinator.get_ranking_by_mu()
    bs_battles = executor.battles_executed
    battled_ids = coordinator._get_battled_player_ids()

    logger.info(f"二分种子: {bs_elapsed:.2f}s, {bs_battles}局, fallback={coordinator.is_fallback}")
    logger.info(f"已对战个体数 m: {len(battled_ids)} / {len(individuals)}")
    logger.info(f"种子: {coordinator.seeds}")
    stats = coordinator.stats
    logger.info(f"统计: launched={stats['candidates_launched']}, "
                 f"promoted={stats['candidates_promoted']}, "
                 f"failed={stats['candidates_failed']}")
    # 对局分类统计（三类）
    bs_no = stats['battles_no_seed']
    bs_with = stats['battles_with_seed']
    bs_final_no = stats['battles_finally_no_seed']
    bs_total = bs_no + bs_with
    if bs_total > 0:
        logger.info(f"对局分类（按对局开始时刻判定，类3=类1子集）:")
        logger.info(f"  类1 [开始时双方非种子]:    {bs_no:5d} ({100*bs_no/bs_total:5.1f}%)")
        logger.info(f"  类2 [开始时一方为种子]:    {bs_with:5d} ({100*bs_with/bs_total:5.1f}%)")
        logger.info(f"  类3 [bisection结束仍非种子]:{bs_final_no:5d} ({100*bs_final_no/bs_total:5.1f}%)")
    # 失败计数
    logger.info(f"失败候选: 凑齐种子前={stats['failed_before_all_seeds_found']}, "
                 f"全程={stats['failed_total']}")

    # Step 4: 全量 RR
    rr_ranking = None
    rr_battles = None
    rr_elapsed = 0.0
    if run_full_rr and n_population <= 80:
        logger.info("── 全量 RoundRobin ──")
        rr_payoff = SimplePayoff()
        t0 = time.time()
        rr_payoff, rr_ranking, rr_battles = run_round_robin(
            individuals=ind_map,
            participant_ids=[ind.id for ind in individuals],
            n_battles=4,
            payoff=rr_payoff,
            seed=seed + 1000,
        )
        rr_elapsed = time.time() - t0
        logger.info(f"RR: {rr_elapsed:.2f}s, {rr_battles}局")

    # Step 5: 指标
    metrics = compute_all_metrics(
        ground_truth, bisection_ranking, rr_ranking,
        battled_ids=battled_ids,
    )
    logger.info(f"\n{format_metrics(metrics)}")

    return {
        "scenario": scenario_name,
        "config": {
            "n_population": n_population, "p_dist": p_dist,
            "q_mode": q_mode, "q_value": q_value,
            "bisection_config": bisection_config, "seed": seed,
        },
        "n_population": n_population,
        "metrics": metrics,
        "total_battles_bisection": bs_battles,
        "total_battles_rr": rr_battles,
        "bisection_elapsed_s": round(bs_elapsed, 3),
        "rr_elapsed_s": round(rr_elapsed, 3),
        "bisection_stats": coordinator.stats,
        "seed_ids": coordinator.seeds,
        "fallback_triggered": coordinator.is_fallback,
        "ground_truth_top10": ground_truth[:10],
        "bisection_top10": bisection_ranking[:10],
        "round_robin_top10": rr_ranking[:10] if rr_ranking else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="二分种子验证实验 (server)")
    parser.add_argument("--preset", choices=["quick", "default", "n80", "full"], default="default")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--p-dist", choices=["uniform", "normal", "bimodal"], default=None)
    parser.add_argument("--q-mode", choices=["constant", "proportional", "random"], default=None)
    parser.add_argument("--q-value", type=float, default=None)
    parser.add_argument("--full-rr", action="store_true", default=True)
    parser.add_argument("--no-full-rr", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # 日志设置
    os.makedirs(args.output_dir, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.output_dir, f"experiment_{ts}.log")
    logger.add(log_file, level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
               encoding="utf-8")
    logger.info(f"日志: {log_file}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Host: {os.uname().nodename}")

    run_rr = args.full_rr and not args.no_full_rr

    # 单场景 or 批次
    if any(v is not None for v in [args.n, args.p_dist, args.q_mode, args.q_value, args.seed]):
        base_config = {"P": 5, "M": 20, "N": 10, "max_seeds": 3,
                        "n_battle": 2, "max_consecutive_fails": 3, "max_workers": 25}
        result = run_single(
            n_population=args.n or 80,
            p_dist=args.p_dist or "uniform",
            q_mode=args.q_mode or "constant",
            q_value=args.q_value or 8.33,
            bisection_config=base_config,
            run_full_rr=run_rr,
            seed=args.seed or 42,
            scenario_name="custom",
        )
        result_file = os.path.join(args.output_dir, f"{ts}_custom.json")
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # 控制台汇总
        m = result["metrics"]
        logger.info(f"\n{'='*60}")
        logger.info(f"ρ_vs_GT={m.get('spearman_rho_vs_ground_truth', 0):.4f}  "
                     f"ρ_vs_RR={m.get('spearman_rho_vs_round_robin', 0):.4f}  "
                     f"T8={m.get('top8_overlap_vs_ground_truth', 0):.4f}  "
                     f"T16={m.get('top16_overlap_vs_ground_truth', 0):.4f}  "
                     f"局数={result['total_battles_bisection']}")
        logger.info(f"结果: {result_file}")
    else:
        scenarios = get_preset_scenarios(args.preset)
        logger.info(f"批次模式: {args.preset} = {len(scenarios)} 场景")
        results = []
        for i, sc in enumerate(scenarios):
            logger.info(f"\n{'#'*60}")
            logger.info(f"场景 [{i+1}/{len(scenarios)}]: {sc['name']}")
            logger.info(f"{'#'*60}")
            r = run_single(
                n_population=sc["n_population"],
                p_dist=sc["p_dist"],
                q_mode=sc["q_mode"],
                q_value=sc["q_value"],
                bisection_config=sc["bisection_config"],
                run_full_rr=run_rr,
                seed=sc.get("seed", 42 + i),
                scenario_name=sc["name"],
            )
            results.append(r)
            # 逐场景保存
            sf = os.path.join(args.output_dir, f"{ts}_{sc['name']}.json")
            with open(sf, "w") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)

        # 汇总
        summary = {"timestamp": datetime.now().isoformat(), "n_scenarios": len(results), "results": [
            {k: v for k, v in r.items() if k != "config"} for r in results
        ]}
        sf = os.path.join(args.output_dir, f"{ts}_summary.json")
        with open(sf, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        latest = os.path.join(args.output_dir, "latest_summary.json")
        with open(latest, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 控制台汇总表
        logger.info(f"\n{'='*92}")
        logger.info(f"{'场景':<25s} {'N':>4s} {'ρ_GT':>8s} {'ρ_RR':>8s} {'Recall8':>8s} {'Recall16':>9s} {'局数':>6s} {'FB':>5s}")
        logger.info("-" * 92)
        for r in results:
            m = r["metrics"]
            logger.info(f"{r['scenario']:<25s} {r['n_population']:>4d} "
                         f"{m.get('spearman_rho_vs_ground_truth',0):>8.4f} "
                         f"{m.get('spearman_rho_vs_round_robin',0):>8.4f} "
                         f"{m.get('recall8_within_top16_vs_ground_truth',0):>8.4f} "
                         f"{m.get('recall16_within_top16_vs_ground_truth',0):>9.4f} "
                         f"{r['total_battles_bisection']:>6d} "
                         f"{str(r['fallback_triggered']):>5s}")
        logger.info(f"{'='*92}")

    logger.info(f"\n完成! 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
