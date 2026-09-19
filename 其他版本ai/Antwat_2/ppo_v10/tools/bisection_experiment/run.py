#!/usr/bin/env python3
"""二分种子方案验证实验 — CLI 入口

用法:
    # 默认 5 场景快速验证
    python -m ppo_v10.tools.bisection_experiment.run

    # 完整 10 场景验证
    python -m ppo_v10.tools.bisection_experiment.run --preset full

    # 自定义
    python -m ppo_v10.tools.bisection_experiment.run --n 80 --p-dist uniform --q-value 10 --full-rr

    # 指定输出目录
    python -m ppo_v10.tools.bisection_experiment.run --output-dir /path/to/results
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from loguru import logger

try:
    from .experiment import run_single_experiment, run_batch_experiments
    from .individual import generate_individuals
except ImportError:
    from experiment import run_single_experiment, run_batch_experiments  # type: ignore
    from individual import generate_individuals  # type: ignore


def get_preset_scenarios(preset: str) -> list:
    """预设场景列表"""

    # 基础二分种子配置
    base_config = {
        "P": 5,
        "M": 20,
        "N": 10,
        "max_seeds": 3,
        "n_battle": 2,
        "max_consecutive_fails": 3,
        "max_workers": 25,
    }

    if preset == "quick":
        # 快速验证: 仅 N=40, uniform
        return [
            {"name": "quick_N40_u8", "n_population": 40, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": base_config, "seed": 42},
        ]

    elif preset == "default":
        # 默认: 覆盖不同 N 和不同 p 分布
        return [
            {"name": "N40_uniform", "n_population": 40, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": base_config, "seed": 42},
            {"name": "N80_uniform", "n_population": 80, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": base_config, "seed": 43},
            {"name": "N80_normal", "n_population": 80, "p_dist": "normal",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": base_config, "seed": 44},
            {"name": "N80_bimodal", "n_population": 80, "p_dist": "bimodal",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": base_config, "seed": 45},
            {"name": "N20_uniform", "n_population": 20, "p_dist": "uniform",
             "q_mode": "constant", "q_value": 8.33, "bisection_config": base_config, "seed": 46},
        ]

    elif preset == "full":
        # 完整验证: 多种 N × 多种分布 × 多种 q 模式
        scenarios = []

        # 不同 N
        for n in [20, 40, 80]:
            for p_dist in ["uniform", "normal", "bimodal"]:
                name = f"N{n}_{p_dist}"
                scenarios.append({
                    "name": name,
                    "n_population": n,
                    "p_dist": p_dist,
                    "q_mode": "constant",
                    "q_value": 8.33,
                    "bisection_config": base_config,
                    "seed": 42 + len(scenarios),
                })

        # 不同 q 模式 (仅 N=80)
        for q_mode in ["proportional", "random"]:
            for q_value in [5.0, 12.0]:
                name = f"N80_uniform_q{q_mode}_{q_value}"
                scenarios.append({
                    "name": name,
                    "n_population": 80,
                    "p_dist": "uniform",
                    "q_mode": q_mode,
                    "q_value": q_value,
                    "bisection_config": base_config,
                    "seed": 100 + len(scenarios),
                })

        return scenarios

    else:
        raise ValueError(f"Unknown preset: {preset}")


def setup_logging(output_dir: str) -> None:
    """配置 loguru 日志：同时输出到控制台和文件"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")

    # 移除默认 handler
    logger.remove()

    # 控制台: INFO
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    # 文件: DEBUG (完整日志)
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        level="DEBUG",
        rotation="100 MB",
        retention="7 days",
        encoding="utf-8",
    )

    logger.info(f"日志文件: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="二分种子方案验证实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--preset",
        choices=["quick", "default", "full"],
        default="default",
        help="预设场景: quick=快速验证, default=5场景, full=完整11场景 (default: default)",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="单个实验的个体数 (覆盖 preset)",
    )
    parser.add_argument(
        "--p-dist", choices=["uniform", "normal", "bimodal"], default=None,
        help="p 分布 (覆盖 preset)",
    )
    parser.add_argument(
        "--q-mode", choices=["constant", "proportional", "random"], default=None,
        help="q 模式 (覆盖 preset)",
    )
    parser.add_argument(
        "--q-value", type=float, default=None,
        help="q 基础值 (覆盖 preset)",
    )
    parser.add_argument(
        "--full-rr", action="store_true", default=True,
        help="运行全量 RoundRobin 对比 (default: True)",
    )
    parser.add_argument(
        "--no-full-rr", action="store_true",
        help="跳过全量 RR (仅 N<=80 适用)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="ppo_v10/outputs/bisection_experiment",
        help="输出目录 (default: ppo_v10/outputs/bisection_experiment)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子",
    )
    parser.add_argument(
        "--bisection-M", type=int, default=None,
        help="覆盖 M 参数 (升级门槛)",
    )
    parser.add_argument(
        "--bisection-N", type=int, default=None,
        help="覆盖 N 参数 (淘汰门槛)",
    )

    args = parser.parse_args()

    # 输出目录
    output_dir = os.path.abspath(args.output_dir)
    setup_logging(output_dir)

    run_rr = args.full_rr and not args.no_full_rr

    # 自定义 bisection_config
    custom_config = None
    if args.bisection_M is not None or args.bisection_N is not None:
        custom_config = {
            "P": 5,
            "M": args.bisection_M or 20,
            "N": args.bisection_N or 10,
            "max_seeds": 3,
            "n_battle": 2,
            "max_consecutive_fails": 3,
            "max_workers": 25,
        }

    # 如果指定了 --n / --p-dist 等参数 → 单场景模式
    if any(v is not None for v in [args.n, args.p_dist, args.q_mode, args.q_value, args.seed]):
        logger.info("单场景模式")
        result = run_single_experiment(
            n_population=args.n or 80,
            p_dist=args.p_dist or "uniform",
            q_mode=args.q_mode or "constant",
            q_value=args.q_value or 8.33,
            bisection_config=custom_config,
            run_full_rr=run_rr,
            seed=args.seed or 42,
            scenario_name="custom",
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"{timestamp}_custom.json")
        try:
            from .experiment import save_result, save_summary
        except ImportError:
            from experiment import save_result, save_summary  # type: ignore
        save_result(result, result_file)
        save_summary([result], os.path.join(output_dir, "latest_summary.json"))

        # 打印最终汇总
        logger.info(f"\n{'='*60}")
        logger.info(f"实验完成!")
        logger.info(f"结果: {result_file}")
        logger.info(f"日志: {output_dir}")
        logger.info(f"{'='*60}")

    else:
        # 批次场景模式
        scenarios = get_preset_scenarios(args.preset)
        if custom_config:
            for sc in scenarios:
                sc["bisection_config"] = custom_config

        logger.info(f"批次模式: {args.preset} = {len(scenarios)} 个场景")
        logger.info(f"全量 RR: {run_rr}")
        logger.info(f"输出目录: {output_dir}")

        results = run_batch_experiments(
            scenarios=scenarios,
            output_dir=output_dir,
            run_full_rr=run_rr,
        )

        # 打印最终汇总表
        logger.info(f"\n{'='*80}")
        logger.info(f"汇总表")
        logger.info(f"{'='*80}")
        header = f"{'场景':<25s} {'N':>4s} {'ρ_vs_GT':>8s} {'ρ_vs_RR':>8s} {'Top8':>7s} {'Top16':>7s} {'局数':>6s} {'Fallback':>8s}"
        logger.info(header)
        logger.info("-" * 80)
        for r in results:
            m = r.metrics
            row = (f"{r.scenario:<25s} {r.n_population:>4d} "
                   f"{m.get('spearman_rho_vs_ground_truth', 0):>8.4f} "
                   f"{m.get('spearman_rho_vs_round_robin', 0):>8.4f} "
                   f"{m.get('top8_overlap_vs_ground_truth', 0):>7.4f} "
                   f"{m.get('top16_overlap_vs_ground_truth', 0):>7.4f} "
                   f"{r.total_battles_bisection:>6d} "
                   f"{str(r.fallback_triggered):>8s}")
            logger.info(row)
        logger.info(f"{'='*80}")
        logger.info(f"完整报告: {output_dir}/latest_summary.json")
        logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
