from dataclasses import dataclass
from typing import List, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from loguru import logger

from ..evolution.population_manager import Individual
from ..battle.agent_loader import AgentLoader
from ..battle.ga_war_agent import GAWarAgent
from ..battle.battle_simulator import _ensure_compatible, _execute_battle
from ..battle.result_aggregator import save_round_logs
from ..genome.genome import load_genome_to_network, cosine_similarity
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..league.battle_shared_payoff import BattleSharedPayoff

_BATTLE_TIMEOUT = 600
_WORKER_DIAG_LOG = "/tmp/ppo_v10_worker_diag.log"
_PROGRESS_INTERVAL = 30  # 每完成 N 个配对输出一次进度日志


def _reconfigure_loguru():
    """在 spawn 子进程中重新配置 loguru sink。"""
    from loguru import logger as _worker_logger
    _worker_logger.remove()
    _worker_logger.add(
        _WORKER_DIAG_LOG,
        level="WARNING",
        rotation="10 MB",
        retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )


def _rr_battle_worker(
    agent_a_bytes: bytes,
    agent_b_bytes: bytes,
    n_battles: int,
    max_rounds: int,
) -> list:
    """子进程中执行两个个体间的对战。

    n_battles 表示"对战次数"，每次对战包含 2 局（先后手各 1 局），
    实际产生 n_battles * 2 局对局。
    """
    _reconfigure_loguru()
    agent_a = AgentLoader.deserialize(agent_a_bytes)
    agent_b = AgentLoader.deserialize(agent_b_bytes)
    agent_a = _ensure_compatible(agent_a)
    agent_b = _ensure_compatible(agent_b)

    results = []
    total_games = n_battles * 2
    for i in range(total_games):
        first_player = i % 2  # 0: a 先手, 1: a 后手
        result = _execute_battle(agent_a, agent_b, first_player, max_rounds)
        result["_agent_a_id"] = getattr(agent_a, "name", None)
        result["_agent_b_id"] = getattr(agent_b, "name", None)
        results.append(result)
    return results


@dataclass
class RoundRobinConfig:
    """循环赛配置"""
    top_n: int = 24
    n_battles: int = 4
    max_workers: int = 25
    max_rounds: int = 512
    device: str = "cuda"
    hidden_dim: int = 256


class RoundRobinTournament:
    """循环赛 — 全圆桌对战 + TrueSkill 排名 + 多样性种子选拔"""

    def __init__(
        self,
        config: RoundRobinConfig,
        payoff: BattleSharedPayoff,
    ):
        self.config = config
        self._payoff = payoff
        self._head_to_head_results: Dict[str, dict] = {}
        # per-pair 结果：pairwise_scores[A][B] = {"wins": ..., "losses": ..., "draws": ...}
        # 表示从 A 的视角对 B 的对战结果
        self._pairwise_scores: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._illegal_a1_total = 0
        self._illegal_a2_total = 0
        self._total_rounds = 0
        self._n_battles = 0

    @property
    def head_to_head_results(self) -> Dict[str, dict]:
        return self._head_to_head_results

    @property
    def pairwise_scores(self) -> Dict[str, Dict[str, Dict[str, int]]]:
        """per-pair 对战结果（A 对 B 视角）。可用于计算单个个体的 Kendall τ。"""
        return self._pairwise_scores

    @property
    def illegal_stats(self) -> Dict[str, int]:
        """返回 round_robin 阶段的非法动作汇总统计。"""
        return {
            "total_illegal": self._illegal_a1_total + self._illegal_a2_total,
            "a1_illegal": self._illegal_a1_total,
            "a2_illegal": self._illegal_a2_total,
            "total_rounds": self._total_rounds,
            "avg_rounds": round(self._total_rounds / max(self._n_battles, 1), 1),
        }

    def _ensure_pair(self, id_a: str, id_b: str) -> None:
        if id_a not in self._pairwise_scores:
            self._pairwise_scores[id_a] = {}
        if id_b not in self._pairwise_scores[id_a]:
            self._pairwise_scores[id_a][id_b] = {"wins": 0, "losses": 0, "draws": 0}
        if id_b not in self._pairwise_scores:
            self._pairwise_scores[id_b] = {}
        if id_a not in self._pairwise_scores[id_b]:
            self._pairwise_scores[id_b][id_a] = {"wins": 0, "losses": 0, "draws": 0}

    def run(
        self,
        top_individuals: List[Individual],
        num_seeds: int,
        elitism_count: int,
        dedup_config: dict,
        generation: int = 1,
        gen_dir: str = None,
    ) -> Tuple[List[Individual], List[Individual]]:
        """
        执行循环赛，返回 (seeds, elites)。

        流程：
        1. Top N 个体全圆桌对战（每对 n_battle 局，先后手各半）
        2. 更新 payoff TrueSkill
        3. 回写 individual.trueskill_mu / trueskill_sigma
        4. 按 mu 降序排列
        5. 基因相似度过滤 → select_seeds_diverse()
        6. 前 elitism_count 个标记为 elites

        Args:
            top_individuals: 基准测试 Top N 的个体（已含 trueskill_mu）
            num_seeds: 种子数量
            elitism_count: 精英数量
            dedup_config: 去重配置 {"hard_threshold": 0.95, "soft_threshold": 0.85, "soft_max_per_family": 2}
            gen_dir: 代次输出目录（用于保存回合日志），为 None 时不保存。

        Returns:
            (seeds: List[Individual], elites: List[Individual])
        """
        n = len(top_individuals)
        if n < 2:
            logger.warning(
                f"[phase=round_robin][gen={generation}] skip: "
                f"reason=too_few_individuals, individuals={n}"
            )
            # 回退：直接按 mu 排序选拔
            ranked = sorted(top_individuals, key=lambda ind: ind.trueskill_mu, reverse=True)
            seeds = self.select_seeds_diverse(
                ranked,
                hard_threshold=dedup_config.get("hard_threshold", 0.95),
                soft_threshold=dedup_config.get("soft_threshold", 0.85),
                soft_max_per_family=dedup_config.get("soft_max_per_family", 2),
                num_seeds=num_seeds,
            )
            elites = seeds[:min(elitism_count, len(seeds))]
            return seeds, elites

        logger.info(
            f"[phase=round_robin][gen={generation}] start: "
            f"individuals={n}, pairs={n * (n-1) // 2}, "
            f"n_battles={self.config.n_battles}, "
            f"games={(n * (n-1) // 2) * self.config.n_battles * 2}, "
            f"max_workers={self.config.max_workers}, top_n={self.config.top_n}"
        )

        # 确保所有个体在 payoff 中注册
        for ind in top_individuals:
            self._payoff.ensure_player(ind.id)

        # 生成对战配对
        matchups = self._generate_matchups(top_individuals)

        # 初始化 head-to-head 结果与 per-pair 结果
        self._head_to_head_results = {}
        self._pairwise_scores = {}
        self._illegal_a1_total = 0
        self._illegal_a2_total = 0
        self._total_rounds = 0
        self._n_battles = 0
        for ind in top_individuals:
            self._head_to_head_results[ind.id] = {"wins": 0, "losses": 0, "draws": 0}

        # 并行执行所有对战
        n_battles = self.config.n_battles
        max_rounds = self.config.max_rounds

        # 预序列化所有 agent
        agent_bytes: Dict[str, bytes] = {}
        for ind in top_individuals:
            ga_agent = self._create_ga_agent(ind)
            agent_bytes[ind.id] = AgentLoader.serialize(ga_agent)

        # 收集所有对战结果用于保存回合日志
        all_battle_results: List[Dict] = []

        max_workers = min(len(matchups), self.config.max_workers)
        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
            futures = {}
            for (ind_a, ind_b) in matchups:
                future = executor.submit(
                    _rr_battle_worker,
                    agent_bytes[ind_a.id],
                    agent_bytes[ind_b.id],
                    n_battles,
                    max_rounds,
                )
                futures[future] = (ind_a.id, ind_b.id)

            completed = 0
            for future in as_completed(futures):
                id_a, id_b = futures[future]
                self._ensure_pair(id_a, id_b)
                try:
                    # n_battles 次对战 = n_battles * 2 局
                    battle_results = future.result(timeout=n_battles * 2 * _BATTLE_TIMEOUT)
                except Exception as e:
                    logger.error(
                        f"[phase=round_robin][gen={generation}] battle_failed: "
                        f"a={id_a}, b={id_b}, error={e}"
                    )
                    # 视为平局：n_battles * 2 局，每局都计平局
                    total_games = n_battles * 2
                    for _ in range(total_games):
                        self._payoff.update(id_a, id_b, 0)
                    self._head_to_head_results[id_a]["draws"] += total_games
                    self._head_to_head_results[id_b]["draws"] += total_games
                    self._pairwise_scores[id_a][id_b]["draws"] += total_games
                    self._pairwise_scores[id_b][id_a]["draws"] += total_games
                    # 记录异常对局
                    for _ in range(total_games):
                        all_battle_results.append({
                            "home": id_a, "away": id_b,
                            "error": f"battle_failed: {e}"
                        })
                    completed += 1
                    continue

                for result in battle_results:
                    # 注入 home/away 用于日志持久化
                    result["home"] = id_a
                    result["away"] = id_b
                    all_battle_results.append(result)

                    if "error" in result:
                        self._payoff.update(id_a, id_b, 0)
                        self._head_to_head_results[id_a]["draws"] += 1
                        self._head_to_head_results[id_b]["draws"] += 1
                        self._pairwise_scores[id_a][id_b]["draws"] += 1
                        self._pairwise_scores[id_b][id_a]["draws"] += 1
                        continue

                    game_result = result.get("result")
                    # 累积非法动作计数
                    self._illegal_a1_total += result.get("illegal_actions_agent1", 0)
                    self._illegal_a2_total += result.get("illegal_actions_agent2", 0)
                    self._total_rounds += result.get("total_rounds", 0)
                    if game_result == "agent1_win":
                        self._payoff.update(id_a, id_b, 1)
                        self._head_to_head_results[id_a]["wins"] += 1
                        self._head_to_head_results[id_b]["losses"] += 1
                        self._pairwise_scores[id_a][id_b]["wins"] += 1
                        self._pairwise_scores[id_b][id_a]["losses"] += 1
                    elif game_result == "agent2_win":
                        self._payoff.update(id_a, id_b, -1)
                        self._head_to_head_results[id_a]["losses"] += 1
                        self._head_to_head_results[id_b]["wins"] += 1
                        self._pairwise_scores[id_a][id_b]["losses"] += 1
                        self._pairwise_scores[id_b][id_a]["wins"] += 1
                    else:
                        self._payoff.update(id_a, id_b, 0)
                        self._head_to_head_results[id_a]["draws"] += 1
                        self._head_to_head_results[id_b]["draws"] += 1
                        self._pairwise_scores[id_a][id_b]["draws"] += 1
                        self._pairwise_scores[id_b][id_a]["draws"] += 1

                completed += 1

                if completed % _PROGRESS_INTERVAL == 0 or completed == len(matchups):
                    logger.info(
                        f"[phase=round_robin][gen={generation}] progress: "
                        f"matchups_completed={completed}/{len(matchups)}"
                    )

        logger.info(
            f"[phase=round_robin][gen={generation}] done: "
            f"matchups_completed={completed}/{len(matchups)}"
        )

        # 保存回合日志
        if gen_dir is not None and all_battle_results:
            save_round_logs("round_robin", generation, gen_dir, all_battle_results)

        # 回写 trueskill_mu / trueskill_sigma
        individual_ids = [ind.id for ind in top_individuals]
        mu_dict = self._payoff.get_trueskill_mus(individual_ids)
        sigma_dict = self._payoff.get_trueskill_sigmas(individual_ids)
        for ind in top_individuals:
            ind.trueskill_mu = mu_dict.get(ind.id, 25.0)
            ind.trueskill_sigma = sigma_dict.get(ind.id, 8.333)

        # 按 mu 降序排列
        ranked = sorted(top_individuals, key=lambda ind: ind.trueskill_mu, reverse=True)

        # 基因相似度过滤选出 seeds
        seeds = self.select_seeds_diverse(
            ranked,
            hard_threshold=dedup_config.get("hard_threshold", 0.95),
            soft_threshold=dedup_config.get("soft_threshold", 0.85),
            soft_max_per_family=dedup_config.get("soft_max_per_family", 2),
            num_seeds=num_seeds,
        )

        # 前 elitism_count 个标记为 elites
        elites = seeds[:min(elitism_count, len(seeds))]

        mu_values = [ind.trueskill_mu for ind in seeds]
        best_mu = mu_values[0] if mu_values else 0.0
        # 总对战数 = sum of all games across all matchups
        self._n_battles = sum(
            v.get("wins", 0) + v.get("losses", 0) + v.get("draws", 0)
            for v in self._head_to_head_results.values()
        ) // 2
        avg_rounds = round(self._total_rounds / max(self._n_battles, 1), 1)
        logger.info(
            f"[phase=round_robin][gen={generation}] done: "
            f"seeds={len(seeds)}, elites={len(elites)}, "
            f"best_mu={best_mu:.2f}, avg_rounds={avg_rounds}"
        )
        illegal_total = self._illegal_a1_total + self._illegal_a2_total
        if illegal_total > 0:
            logger.info(
                f"[phase=round_robin][gen={generation}] illegal: "
                f"total={illegal_total}, "
                f"a1_illegal={self._illegal_a1_total}, "
                f"a2_illegal={self._illegal_a2_total}, "
                f"avg_rounds={avg_rounds}"
            )

        return seeds, elites

    def _generate_matchups(
        self,
        individuals: List[Individual],
    ) -> List[Tuple[Individual, Individual]]:
        """生成全圆桌对战配对列表（每对只出现一次）"""
        matchups = []
        n = len(individuals)
        for i in range(n):
            for j in range(i + 1, n):
                matchups.append((individuals[i], individuals[j]))
        return matchups

    def select_seeds_diverse(
        self,
        ranked: List[Individual],
        hard_threshold: float = 0.95,
        soft_threshold: float = 0.85,
        soft_max_per_family: int = 2,
        num_seeds: int = 8,
    ) -> List[Individual]:
        """
        基因相似度去重选出 seeds。
        与 v9 BattleSelection.select_seeds_diverse() 完全相同的逻辑，
        仅将 ELO 排序改为按 trueskill_mu 排序。

        Args:
            ranked: 按 trueskill_mu 降序排列的个体列表
            hard_threshold: 硬去重的余弦相似度阈值（默认 0.95）
            soft_threshold: 同一家族的余弦阈值（默认 0.85）
            soft_max_per_family: 每家族最大名额（默认 2）
            num_seeds: 目标种子数量

        Returns:
            去重后的种子列表
        """
        selected: List[Individual] = []
        selected_weights: List[np.ndarray] = []

        for candidate in ranked:
            w_candidate = candidate.genome.weights

            # 检查是否与已选种子硬重复
            is_hard_dup = False
            family_count = 0
            for sel_w in selected_weights:
                cos = cosine_similarity(w_candidate, sel_w)

                if cos > hard_threshold:
                    is_hard_dup = True
                    logger.debug(
                        f"Hard duplicate: {candidate.id} (mu={candidate.trueskill_mu:.2f}) "
                        f"cos={cos:.4f}"
                    )
                    break

                if cos > soft_threshold:
                    family_count += 1

            if is_hard_dup:
                continue

            if family_count >= soft_max_per_family:
                logger.debug(
                    f"Soft throttled: {candidate.id} (mu={candidate.trueskill_mu:.2f}) "
                    f"family already has {family_count} seeds"
                )
                continue

            # 通过筛选
            selected.append(candidate)
            selected_weights.append(w_candidate)
            if len(selected) >= num_seeds:
                break

        # ── 兜底：不足 num_seeds，逐级放宽 ──
        if len(selected) < num_seeds:
            selected_ids = {ind.id for ind in selected}
            remaining = [ind for ind in ranked if ind.id not in selected_ids]

            for fallback_level, (st, sm) in enumerate([
                (soft_threshold + 0.05, 3),   # Level 1
                (soft_threshold + 0.05, 4),   # Level 2
                (1.0, num_seeds),              # Level 3: 完全降级
            ]):
                for candidate in remaining:
                    if candidate.id in selected_ids:
                        continue
                    if len(selected) >= num_seeds:
                        break

                    w_candidate = candidate.genome.weights
                    family_count = 0
                    is_hard = False
                    for sel_w in selected_weights:
                        cos = cosine_similarity(w_candidate, sel_w)
                        if cos > hard_threshold:
                            is_hard = True
                            break
                        if cos > st:
                            family_count += 1
                    if is_hard:
                        continue
                    if family_count >= sm:
                        continue

                    selected.append(candidate)
                    selected_weights.append(w_candidate)
                    selected_ids.add(candidate.id)

                if len(selected) >= num_seeds:
                    logger.warning(
                        f"Seed dedup fallback level {fallback_level}: "
                        f"got {len(selected)} seeds"
                    )
                    break

        # 最终兜底：仍然不够，直接按 mu 取
        if len(selected) < num_seeds:
            logger.error(
                f"Seed dedup exhausted: only {len(selected)}/{num_seeds} seeds selected. "
                f"Using mu-only fallback."
            )
            selected = ranked[:num_seeds]

        # 设置种子排名
        for i, seed in enumerate(selected):
            seed.seed_rank = i + 1

        # 日志：输出种子间的多样性概览
        if len(selected) >= 2:
            cos_vals = []
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    cos_vals.append(cosine_similarity(
                        selected_weights[i],
                        selected_weights[j],
                    ))
            logger.info(
                f"Seeds diversity: {len(selected)} seeds, "
                f"cos range=[{min(cos_vals):.3f}, {max(cos_vals):.3f}], "
                f"cos mean={sum(cos_vals) / len(cos_vals):.3f}"
            )

        return selected[:num_seeds]

    def _create_ga_agent(self, individual: Individual, player_id: int = 0) -> GAWarAgent:
        """从 Individual 创建 GAWarAgent（用于对战）。"""
        import torch
        device = torch.device(self.config.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.config.hidden_dim,
        ).to(device)
        load_genome_to_network(network, individual.genome)
        network.eval()
        return GAWarAgent(
            player_id=player_id,
            network=network,
            device=device,
        )
