#!/usr/bin/env python3
"""Gen 5 种子 vs 外部 baseline agent 对战评估。

用法（在 server 上）:
    python tools/eval_seeds_vs_baselines.py \
        --seed_dir outputs/<run_id>/generations/gen_0005/seed_models \
        --agents sample,nn_gen_32_cuda,rule_zzy25 \
        --n_battles 25 \
        --max_workers 8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ── 确保 SDK 和 ppo_v9 源码可导入 ──
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
SDK_DIR = PROJECT_DIR.parent / "Ant-Game"  # SDK 路径
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SDK_DIR))

from ppo_ga.battle.ga_war_agent import GAWarAgent
from ppo_ga.battle.agent_loader import AgentLoader
from ppo_ga.battle.battle_simulator import BattleSimulator
from ppo_ga.battle.battle_config import BaselineBattleConfig
from ppo_ga.battle.result_aggregator import aggregate_results
from ppo_ga.network.ant_war_policy_value_network import AntWarPolicyValueNetwork

# 设置 baselines 目录的绝对路径
BASELINES_DIR = Path(__file__).resolve().parent.parent.parent / "baselines"
AgentLoader.set_baselines_path(str(BASELINES_DIR))


def load_seed_network(pt_path: str, device: torch.device, hidden_dim: int = 256, enable_auxiliary: bool = True):
    """从种子 .pt 文件加载网络。

    种子文件格式: {"policy_state_dict": {...}, "individual_id": ..., ...}
    """
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    if "policy_state_dict" in data:
        state_dict = data["policy_state_dict"]
    elif "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        state_dict = data

    network = AntWarPolicyValueNetwork(
        hidden_dim=hidden_dim,
        enable_auxiliary=enable_auxiliary,
    ).to(device)
    network.load_state_dict(state_dict, strict=False)
    network.eval()
    return network


def main():
    parser = argparse.ArgumentParser(description="Gen 5 种子 vs 外部 baseline 对战评估")
    parser.add_argument("--seed_dir", required=True, help="种子模型目录")
    parser.add_argument("--agents", default="sample,nn_gen_32_cuda,rule_zzy25",
                        help="逗号分隔的 baseline agent 列表")
    parser.add_argument("--n_battles", type=int, default=25, help="每个 (种子,agent) 组合的对战次数")
    parser.add_argument("--max_workers", type=int, default=8, help="并行工作进程数")
    parser.add_argument("--max_rounds", type=int, default=512, help="每局最大回合数")
    parser.add_argument("--device", default="cuda", help="设备 (cuda/cpu)")
    parser.add_argument("--output_json", default=None, help="结果输出 JSON 文件路径")
    args = parser.parse_args()

    seed_dir = Path(args.seed_dir)
    if not seed_dir.is_dir():
        print(f"ERROR: 种子目录不存在: {seed_dir}")
        sys.exit(1)

    agent_names = [a.strip() for a in args.agents.split(",")]

    # ── 设备 ──
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA 不可用，回退到 CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"设备: {device}")

    # ── 集合种子文件 ──
    seed_files = sorted(
        [f for f in os.listdir(seed_dir) if f.endswith(".pt")],
        key=lambda x: int(x.replace("seed_", "").replace(".pt", ""))
    )
    if not seed_files:
        print(f"ERROR: {seed_dir} 中没有 .pt 文件")
        sys.exit(1)

    print(f"\n种子模型 ({len(seed_files)} 个):")
    for f in seed_files:
        path = seed_dir / f
        meta = torch.load(path, map_location="cpu", weights_only=True)
        print(f"  {f}: id={meta.get('individual_id','?')} "
              f"elo={meta.get('elo_rating', 0):.1f} "
              f"parents={meta.get('parent_ids', [])}")

    # ── 创建 BattleSimulator ──
    config = BaselineBattleConfig(
        max_rounds=args.max_rounds,
        n_battles=args.n_battles,
        max_workers=args.max_workers,
    )
    simulator = BattleSimulator(config)

    print(f"\n对战配置: n_battles={args.n_battles}, 实际每组合 {args.n_battles*2} 局")
    print(f"Baseline agents: {agent_names}")

    # ── 逐种子评估 ──
    all_results = {}
    total_start = time.time()

    for sf in seed_files:
        seed_path = seed_dir / sf
        seed_data = torch.load(seed_path, map_location="cpu", weights_only=True)
        seed_id = seed_data.get("individual_id", sf.replace(".pt", ""))
        seed_elo = seed_data.get("elo_rating", 0)
        seed_rank = seed_data.get("seed_rank", "?")

        print(f"\n{'='*70}")
        print(f"种子 #{seed_rank}: {seed_id} (ELO={seed_elo:.1f})")
        print(f"{'='*70}")

        # 加载网络
        t0 = time.time()
        network = load_seed_network(str(seed_path), device)
        ga_agent = GAWarAgent(player_id=0, network=network, device=device)
        print(f"  网络加载: {(time.time()-t0)*1000:.0f}ms")

        seed_results = {}
        for agent_name in agent_names:
            print(f"  vs {agent_name} ...", end=" ", flush=True)
            t0 = time.time()

            try:
                baseline_agent = AgentLoader.load(agent_name, player_id=1)
                battle_results = simulator.run_battles(
                    ga_agent, baseline_agent, args.n_battles
                )
                agg = aggregate_results(battle_results)

                total_battles = agg.get("total_battles", args.n_battles * 2)
                wins = agg.get("agent1_wins", 0)
                losses = agg.get("agent2_wins", 0)
                draws = agg.get("draws", 0)
                errors = total_battles - wins - losses - draws
                win_rate = wins / max(total_battles, 1)
                elapsed = time.time() - t0

                print(f"win_rate={win_rate:.1%} ({wins}W/{losses}L/{draws}D) "
                      f"[{elapsed:.1f}s]")

                seed_results[agent_name] = {
                    "win_rate": round(win_rate, 4),
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "errors": errors,
                    "total_battles": total_battles,
                    "elapsed_seconds": round(elapsed, 1),
                }
            except Exception as e:
                print(f"ERROR: {e}")
                seed_results[agent_name] = {
                    "win_rate": 0,
                    "error": str(e),
                }

        # 汇总
        wrs = [r.get("win_rate", 0) for r in seed_results.values() if "error" not in r]
        avg_wr = sum(wrs) / max(len(wrs), 1) if wrs else 0
        print(f"  => 平均胜率: {avg_wr:.1%}")

        all_results[seed_id] = {
            "seed_rank": seed_rank,
            "elo_rating": round(seed_elo, 1),
            "avg_win_rate": round(avg_wr, 4),
            "results": seed_results,
        }

        # 释放 GPU 显存
        del network, ga_agent
        torch.cuda.empty_cache()

    # ── 汇总 ──
    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"评估完成，总耗时 {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")

    print(f"\n{'种子':>25}  {'ELO':>8}", end="")
    for an in agent_names:
        print(f"  {an:>12}", end="")
    print(f"  {'平均':>8}")
    print("-" * (44 + 14 * len(agent_names)))

    for seed_id, sr in all_results.items():
        short_id = seed_id.replace("gen_005_", "")
        print(f"  {short_id:>23}  {sr['elo_rating']:8.1f}", end="")
        for an in agent_names:
            wr = sr["results"].get(an, {}).get("win_rate", 0)
            print(f"  {wr:11.1%}", end="")
        print(f"  {sr['avg_win_rate']:7.1%}")

    # ── 保存 ──
    output_data = {
        "config": {
            "seed_dir": str(seed_dir),
            "agents": agent_names,
            "n_battles": args.n_battles,
            "max_rounds": args.max_rounds,
            "device": str(device),
        },
        "results": all_results,
    }

    output_path = args.output_json or f"eval_gen5_vs_baselines_{int(time.time())}.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {output_path}")


if __name__ == "__main__":
    main()
