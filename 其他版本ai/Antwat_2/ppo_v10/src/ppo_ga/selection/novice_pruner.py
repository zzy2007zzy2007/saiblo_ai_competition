"""剪枝器 —— 在 touchstone 前过滤弱个体。

支持按世代切换对手：
- gen1-gen3: BasicRandomAI（降低门槛，保留多样性）
- gen4+:     NoviceAI（提高门槛，严格筛选）

每个个体与对手对战 n_battle 次（每次先后手各一局），
只有取得全胜的个体才能进入下一阶段。
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import Any, Dict, List

import torch
from loguru import logger

from ..battle.agent_loader import AgentLoader
from ..battle.battle_simulator import run_single_battle_process
from ..battle.ga_war_agent import GAWarAgent
from ..battle.result_aggregator import aggregate_illegal_stats, save_round_logs
from ..evolution.population_manager import Individual
from ..genome.genome import load_genome_to_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork

_BATTLE_TIMEOUT_DEFAULT = 600


class NovicePruner:
    """剪枝器 —— 支持按世代切换对手与参数。

    前 early_gens 代使用 early 配置，之后使用 late 配置。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: str = "cpu",
        hidden_dim: int = 256,
        max_rounds: int = 512,
    ):
        self._early_gens = config.get("early_gens", 3)
        self._early_config = config["early"]
        self._late_config = config["late"]
        self._device = torch.device(device)
        self._hidden_dim = hidden_dim
        self._max_rounds = max_rounds

    @property
    def n_battle(self) -> int:
        # 返回 early 的 n_battle（向后兼容，仅用于初始化日志）
        return self._early_config["n_battle"]

    @property
    def early_gens(self) -> int:
        return self._early_gens

    def _get_config(self, generation: int) -> Dict[str, Any]:
        """根据世代选择配置。"""
        if generation <= self._early_gens:
            return self._early_config
        return self._late_config

    def prune(
        self,
        individuals: List[Individual],
        generation: int,
        gen_dir: str = None,
    ) -> tuple:
        """对个体列表执行 NoviceAI 剪枝。

        Args:
            individuals: 待剪枝的个体列表。
            generation: 当前代次（用于日志）。
            gen_dir: 代次输出目录（用于保存回合日志），为 None 时不保存。

        Returns:
            (survivors, stats): survivors 为通过剪枝的个体列表，
            stats 为 {"total": int, "passed": int} 统计信息。
        """
        total = len(individuals)
        if total == 0:
            return [], {"total": 0, "passed": 0}

        cfg = self._get_config(generation)
        opponent_name = cfg["opponent"]
        n_battle = cfg["n_battle"]
        max_workers = cfg["max_workers"]
        battle_timeout = cfg.get("battle_timeout", _BATTLE_TIMEOUT_DEFAULT)

        total_games = total * n_battle * 2
        logger.info(
            f"[phase=pruning][gen={generation}] start: "
            f"individuals={total}, opponent={opponent_name}, "
            f"n_battle={n_battle}, games={total_games}, "
            f"max_workers={max_workers}"
        )

        # 预创建并序列化对手 agent（所有对战共用）
        opponent_agent = AgentLoader.load_builtin(opponent_name, player_id=1)
        opponent_bytes = AgentLoader.serialize(opponent_agent)

        # 为每个个体创建 GAWarAgent 并序列化
        ind_ga_bytes: Dict[str, bytes] = {}
        for ind in individuals:
            network = AntWarPolicyValueNetwork(
                hidden_dim=self._hidden_dim,
            ).to(self._device)
            load_genome_to_network(network, ind.genome)
            network.eval()
            ga_agent = GAWarAgent(player_id=0, network=network, device=self._device)
            ind_ga_bytes[ind.id] = AgentLoader.serialize(ga_agent)

        # 提交所有对战任务到单个进程池
        ind_results: Dict[str, List[dict]] = defaultdict(list)

        spawn_ctx = mp.get_context("spawn")
        pool = ProcessPoolExecutor(
            max_workers=max_workers, mp_context=spawn_ctx
        )
        pool_closed = False
        try:
            futures: dict = {}
            for ind in individuals:
                ga_bytes = ind_ga_bytes[ind.id]
                for game_idx in range(n_battle * 2):
                    first_player = game_idx % 2  # 0: GA先手, 1: GA后手
                    f = pool.submit(
                        run_single_battle_process,
                        ga_bytes,
                        opponent_bytes,
                        first_player,
                        self._max_rounds,
                    )
                    futures[f] = ind.id

            pending = set(futures)
            deadlines = {f: time.monotonic() + battle_timeout for f in futures}
            while pending:
                now = time.monotonic()
                timed_out = [f for f in pending if deadlines[f] <= now]
                if timed_out:
                    timed_out_set = set(timed_out)
                    for f in timed_out:
                        ind_id = futures[f]
                        f.cancel()
                        logger.error(
                            f"[phase=pruning][gen={generation}] battle_timeout: "
                            f"individual={ind_id}, timeout={battle_timeout}s"
                        )
                        ind_results[ind_id].append(
                            {"result": "error", "error": "battle timeout"}
                        )
                    for f in pending - timed_out_set:
                        ind_id = futures[f]
                        f.cancel()
                        ind_results[ind_id].append(
                            {"result": "error", "error": "battle cancelled after timeout"}
                        )
                    terminate_workers = getattr(pool, "terminate_workers", None)
                    if terminate_workers is not None:
                        terminate_workers()
                    else:
                        pool.shutdown(wait=False, cancel_futures=True)
                    pool_closed = True
                    break

                timeout = max(0.0, min(deadlines[f] for f in pending) - time.monotonic())
                done, _ = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
                for f in done:
                    ind_id = futures[f]
                    try:
                        result = f.result()
                    except Exception as e:
                        logger.error(
                            f"[phase=pruning][gen={generation}] battle_failed: "
                            f"individual={ind_id}, error={e}"
                        )
                        result = {"result": "error", "error": str(e)}
                    ind_results[ind_id].append(result)
                    pending.remove(f)
        finally:
            if not pool_closed:
                pool.shutdown(wait=True)

        # 过滤：只有全胜个体通过
        survivors: List[Individual] = []
        passed = 0
        for ind in individuals:
            results = ind_results.get(ind.id, [])
            wins = sum(
                1 for r in results if r.get("result") == "agent1_win"
            )
            if wins == n_battle * 2:
                survivors.append(ind)
                passed += 1
            else:
                logger.info(
                    f"[phase=pruning][gen={generation}] individual_pruned: "
                    f"individual={ind.id}, wins={wins}/{n_battle * 2}"
                )

        stats = {"total": total, "passed": passed}

        # 保存回合日志
        if gen_dir is not None:
            all_results = []
            for ind in individuals:
                for result in ind_results.get(ind.id, []):
                    result["home"] = ind.id
                    result["away"] = opponent_name
                    all_results.append(result)
            save_round_logs("pruning", generation, gen_dir, all_results)

        # 保存摘要报告
        if gen_dir is not None:
            try:
                import json
                import os

                # 逐个体胜场统计
                ind_win_stats = []
                win_dist = {}
                total_individual_rounds = 0
                total_individual_games = 0
                for ind in individuals:
                    results = ind_results.get(ind.id, [])
                    wins = sum(1 for r in results if r.get("result") == "agent1_win")
                    losses = sum(1 for r in results if r.get("result") == "agent2_win")
                    draws = len(results) - wins - losses
                    total_games = len(results)
                    total_rounds = sum(r.get("total_rounds", 0) for r in results)
                    avg_rounds = round(total_rounds / total_games, 1) if total_games > 0 else 0.0
                    total_individual_rounds += total_rounds
                    total_individual_games += total_games
                    ind_win_stats.append({
                        "id": ind.id,
                        "wins": wins,
                        "losses": losses,
                        "draws": draws,
                        "total_games": total_games,
                        "total_rounds": total_rounds,
                        "avg_rounds": avg_rounds,
                        "win_rate": round(wins / total_games, 4) if total_games > 0 else 0.0,
                        "survived": wins == n_battle * 2,
                    })
                    win_dist[wins] = win_dist.get(wins, 0) + 1

                # 非法动作统计
                illegal_stats = aggregate_illegal_stats(all_results) if all_results else {}

                # 聚合平均胜率
                pruned_stats = [s for s in ind_win_stats if not s["survived"]]
                avg_pruned_win_rate = (
                    sum(s["win_rate"] for s in pruned_stats) / len(pruned_stats)
                    if pruned_stats else 0.0
                )
                survived_stats = [s for s in ind_win_stats if s["survived"]]
                avg_survived_win_rate = (
                    sum(s["win_rate"] for s in survived_stats) / len(survived_stats)
                    if survived_stats else 0.0
                )

                # 构建字段——去除不存在的 avg_per_battle/avg_per_round
                overall_avg_rounds = round(total_individual_rounds / total_individual_games, 1) if total_individual_games > 0 else 0.0
                summary = {
                    "phase": "pruning",
                    "generation": generation,
                    "opponent": opponent_name,
                    "n_battle": n_battle,
                    "total_individuals": total,
                    "passed": passed,
                    "pruned": total - passed,
                    "pass_rate": round(passed / total, 4) if total > 0 else 0.0,
                    "avg_rounds": overall_avg_rounds,
                    "avg_pruned_win_rate": round(avg_pruned_win_rate, 4),
                    "avg_survived_win_rate": round(avg_survived_win_rate, 4),
                    "total_illegal": illegal_stats.get("total_illegal", 0),
                    "agent1_illegal": illegal_stats.get("agent1_illegal", 0),
                    "agent2_illegal": illegal_stats.get("agent2_illegal", 0),
                    "win_distribution": dict(sorted(win_dist.items())),
                    "individuals": sorted(ind_win_stats, key=lambda x: x["win_rate"], reverse=True),
                }
                summary_path = os.path.join(gen_dir, "pruning_summary.json")
                with open(summary_path, "w") as f:
                    json.dump(summary, f, indent=2, default=str)

                logger.info(
                    f"[phase=pruning][gen={generation}] summary saved: "
                    f"pass_rate={summary['pass_rate']:.2%}, "
                    f"illegal={summary['total_illegal']}, "
                    f"avg_rounds={summary['avg_rounds']}, "
                    f"survived_avg_win_rate={avg_survived_win_rate:.2%}"
                )
            except Exception as e:
                logger.error(
                    f"[phase=pruning][gen={generation}] summary save failed: {e}"
                )

        if passed == 0:
            logger.warning(
                f"[phase=pruning][gen={generation}] done: "
                f"survivors=0/{total}, all_pruned=true"
            )
        else:
            logger.info(
                f"[phase=pruning][gen={generation}] done: "
                f"survivors={passed}/{total}, all_pruned=false"
            )

        return survivors, stats
