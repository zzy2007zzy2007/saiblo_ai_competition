#!/usr/bin/env python3
"""全体个体 vs 多个 baseline AI 的对战评估。

复用 BaselineEvaluator 的 evaluate_seeds() 代码路径，
与训练中阶段四（基线验证）完全相同的执行模式。

用法（在 server 上）:
    python tools/eval_inds_vs_baselines.py \
        --checkpoint outputs/ga_run_xxx/checkpoint/checkpoint_gen_0005.pt \
        --agents BasicRandomAI,NoviceAI,BasicTowerAI,MediumRuleAI \
        --n_battles 6 \
        --max_workers 25 \
        --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# ── 确保 SDK 和 ppo_v9 源码可导入 ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
SDK_DIR = PROJECT_DIR.parent / "Ant-Game"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SDK_DIR))

from ppo_ga.battle.agent_loader import AgentLoader
from ppo_ga.battle.battle_config import GABattleConfig
from ppo_ga.evaluation.baseline_evaluator import BaselineEvaluator
from ppo_ga.trainer.checkpoint_manager import GACheckpointManager

BASELINES_DIR = PROJECT_DIR.parent / "baselines"
AgentLoader.set_baselines_path(str(BASELINES_DIR))


def compute_stats(values: List[float]) -> Dict[str, float]:
    """计算分布统计量。"""
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="全体个体 vs baseline AI 对战评估（区分度测试）"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="checkpoint .pt 文件路径")
    parser.add_argument("--agents", default="BasicRandomAI,NoviceAI,BasicTowerAI,MediumRuleAI",
                        help="逗号分隔的 baseline AI 名称列表")
    parser.add_argument("--n_battles", type=int, default=6,
                        help="每个(个体, baseline)组合的对战次数（默认 6，每局 2 面共 12 场）")
    parser.add_argument("--max_workers", type=int, default=12,
                        help="并行工作进程数")
    parser.add_argument("--max_rounds", type=int, default=512,
                        help="每局最大回合数")
    parser.add_argument("--device", default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="网络隐藏层维度")
    parser.add_argument("--output_json", default=None,
                        help="结果输出 JSON 文件路径（默认自动生成）")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        print(f"ERROR: checkpoint 文件不存在: {checkpoint_path}")
        sys.exit(1)

    agent_names = [a.strip() for a in args.agents.split(",")]
    print(f"Baseline AI: {agent_names}")

    # ── 设备 ──
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA 不可用，回退到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"设备: {device}")

    # ── 加载 checkpoint ──
    print(f"\n加载 checkpoint: {checkpoint_path}")
    t0 = time.time()
    mgr = GACheckpointManager(device)
    data = mgr.load(
        str(checkpoint_path),
        hidden_dim=args.hidden_dim,
    )
    population = data["population"]
    gen = data["generation"]
    individuals = population.individuals
    print(f"  代次: {gen}")
    print(f"  个体数: {len(individuals)}")
    print(f"  加载耗时: {time.time() - t0:.1f}s")

    if len(individuals) == 0:
        print("ERROR: checkpoint 中无个体数据")
        sys.exit(1)

    # ── 按 ELO 降序 ──
    individuals = sorted(individuals, key=lambda ind: ind.elo_rating, reverse=True)

    # ── 创建 BaselineEvaluator —— 与训练中阶段四相同的代码路径 ──
    config = GABattleConfig(
        max_rounds=args.max_rounds,
        n_battles=args.n_battles,
        max_workers=args.max_workers,
    )
    evaluator = BaselineEvaluator(
        config=config,
        device=str(device),
        hidden_dim=args.hidden_dim,
        n_battles=args.n_battles,
        baseline_agents=agent_names,
        enabled=True,
        interval=1,  # 确保 should_evaluate 始终返回 True
    )

    total_combos = len(individuals) * len(agent_names)
    total_battles = total_combos * args.n_battles * 2
    print(f"\n对战配置:")
    print(f"  n_battles={args.n_battles} → 每组合 {args.n_battles * 2} 局")
    print(f"  {len(individuals)} 个体 × {len(agent_names)} baseline "
          f"= {total_combos} 组合")
    print(f"  总对战局数: {total_battles}")
    print(f"  执行模式: BaselinEvaluator.evaluate_seeds()（与训练阶段四一致）")

    # ── 执行评估 ──
    print(f"\n开始评估...")
    total_start = time.time()
    raw_results = evaluator.evaluate_seeds(individuals, generation=0)
    total_elapsed = time.time() - total_start

    # ── 解析结果 ──
    individual_results: List[Dict] = []
    for rank_idx, ind in enumerate(individuals):
        ind_data = raw_results.get(ind.id, {})
        if not ind_data:
            individual_results.append({
                "id": ind.id,
                "elo_rating": round(ind.elo_rating, 1),
                "error": "no result",
            })
            continue

        wr_map: Dict[str, float] = {}
        detail_map: Dict[str, Dict] = {}
        for baseline_name in agent_names:
            agg = ind_data.get(baseline_name, {})
            if not agg:
                wr_map[baseline_name] = 0.0
                detail_map[baseline_name] = {"error": "no result"}
                continue

            total = agg.get("total_battles", args.n_battles * 2)
            wins = agg.get("agent1_wins", 0)
            losses = agg.get("agent2_wins", 0)
            draws = agg.get("draws", 0)
            wr = wins / max(total, 1)

            wr_map[baseline_name] = wr
            detail_map[baseline_name] = {
                "win_rate": round(wr, 4),
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "total_battles": total,
            }

        avg_wr = sum(wr_map.values()) / max(len(wr_map), 1)
        individual_results.append({
            "id": ind.id,
            "elo_rating": round(ind.elo_rating, 1),
            "seed_rank": ind.seed_rank,
            "win_rates": wr_map,
            "avg_win_rate": round(avg_wr, 4),
            "details": detail_map,
        })

    # ═══════════════════════════════════════════════════════════════════════
    # 汇总报告
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print(f"评估完成，总耗时 {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)")
    print(f"{'=' * 80}")

    # ── 汇总表 ──
    print(f"\n{'个体ID':>28}  {'ELO':>8}", end="")
    for name in agent_names:
        print(f"  {name:>14}", end="")
    print(f"  {'平均':>8}  {'排名':>6}")
    print("-" * (44 + 16 * len(agent_names)))

    for i, r in enumerate(individual_results):
        if "error" in r:
            print(f"  {r['id']:>26}  ERROR: {r['error']}")
            continue
        wrs = r["win_rates"]
        avg = r["avg_win_rate"]
        print(f"  {r['id']:>26}  {r['elo_rating']:8.1f}", end="")
        for name in agent_names:
            wr = wrs.get(name, 0)
            print(f"  {wr:13.1%}", end="")
        print(f"  {avg:7.1%}  {i + 1:>6}")

    # ── 区分度分析 ──
    print(f"\n{'=' * 80}")
    print("区分度分析")
    print(f"{'=' * 80}")

    for baseline_name in agent_names:
        wrs = []
        for r in individual_results:
            if "error" not in r and baseline_name in r.get("win_rates", {}):
                wrs.append(r["win_rates"][baseline_name])

        if not wrs:
            continue

        stats = compute_stats(wrs)
        print(f"\n--- {baseline_name} ---")
        print(f"  胜率范围: [{stats['min']:.1%}, {stats['max']:.1%}]")
        print(f"  均值: {stats['mean']:.1%}")
        print(f"  标准差: {stats['std']:.1%}")
        print(f"  中位数: {stats['median']:.1%}")
        print(f"  IQR (P25-P75): [{stats['p25']:.1%}, {stats['p75']:.1%}]")

    # ── 区分度排名 ──
    print(f"\n--- 区分度排名 (按 std 降序) ---")
    diff_scores = []
    for baseline_name in agent_names:
        wrs = []
        for r in individual_results:
            if "error" not in r and baseline_name in r.get("win_rates", {}):
                wrs.append(r["win_rates"][baseline_name])
        if wrs:
            stats = compute_stats(wrs)
            diff_scores.append((baseline_name, stats["std"], stats["mean"]))

    diff_scores.sort(key=lambda x: x[1], reverse=True)
    print(f"  {'Baseline':>20}  {'Std':>8}  {'Mean':>8}  {'区分度等级':>12}")
    print(f"  {'-' * 52}")
    for name, std, mean in diff_scores:
        if std > 0.15:
            grade = "高"
        elif std > 0.05:
            grade = "中"
        else:
            grade = "低"
        print(f"  {name:>20}  {std:7.1%}  {mean:7.1%}  {grade:>12}")

    # ── 保存 JSON ──
    output_data = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "generation": gen,
            "baseline_agents": agent_names,
            "n_battles": args.n_battles,
            "max_rounds": args.max_rounds,
            "max_workers": args.max_workers,
            "device": str(device),
        },
        "individuals": individual_results,
        "differentiation": {
            name: compute_stats([
                r["win_rates"][name]
                for r in individual_results
                if "error" not in r and name in r.get("win_rates", {})
            ])
            for name in agent_names
        },
    }

    output_path = args.output_json or \
        f"eval_ind_{gen}_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
