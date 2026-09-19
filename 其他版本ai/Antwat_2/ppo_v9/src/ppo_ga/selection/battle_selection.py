from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from loguru import logger

from ..evolution.population_manager import Individual, Population
from ..battle.battle_simulator import BattleSimulator, _ensure_compatible, _execute_battle
from ..battle.battle_config import GABattleConfig
from ..battle.ga_war_agent import GAWarAgent
from ..battle.agent_loader import AgentLoader
from ..genome.genome import load_genome_to_network, cosine_similarity
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from .elo_rating import ELORating

_BATTLE_TIMEOUT = 300  # 单局超时秒数
_WORKER_DIAG_LOG = "/tmp/ppo_v9_worker_diag.log"


def _reconfigure_loguru():
    """在 spawn 子进程中重新配置 loguru sink。
    
    由于 mp.get_context(\"spawn\") 创建的子进程不继承父进程的
    loguru sink，worker 函数必须在入口调用此函数来重新添加文件 sink，
    否则 logger.warning() 等输出会丢失。
    """
    from loguru import logger as _worker_logger
    _worker_logger.remove()  # 移除默认 stderr handler
    _worker_logger.add(
        _WORKER_DIAG_LOG,
        level="WARNING",
        rotation="10 MB",
        retention=3,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    )


# ── 参照物定义 ──────────────────────────────────────────────────────────────

@dataclass
class ReferenceAgent:
    """参照物 Agent 定义"""
    name: str           # 名称
    elo: float          # ELO 评分（锚定参照物为固定值，动态参照物为上一轮实际评分）
    agent_factory: Any  # 创建 Agent 的工厂（player_id -> Agent 实例）
    is_anchor: bool = True  # 是否为锚定固定参照物


# 默认锚定参照物（永不更换）
DEFAULT_ANCHOR_REFERENCES = [
    ReferenceAgent("BasicRandomAI", 400.0, lambda pid: AgentLoader.load("BasicRandomAI", pid), is_anchor=True),
    ReferenceAgent("BasicTowerAI", 1000.0, lambda pid: AgentLoader.load("BasicTowerAI", pid), is_anchor=True),
    ReferenceAgent("MediumRuleAI", 1600.0, lambda pid: AgentLoader.load("MediumRuleAI", pid), is_anchor=True),
]

# agent 名称 → 默认 ELO 评分映射
DEFAULT_ELO_MAP: Dict[str, float] = {
    "BasicRandomAI": 400.0,
    "BasicTowerAI": 1000.0,
    "MediumRuleAI": 1600.0,
    "sample": 1200.0,
}


def build_anchor_references(
    anchor_names: List[str],
    elo_overrides: Optional[Dict[str, float]] = None,
) -> List[ReferenceAgent]:
    """根据配置构建锚定参照物列表。

    Args:
        anchor_names: 参照物名称列表
        elo_overrides: 可选的 ELO 覆盖（{name: elo}）

    Returns:
        参照物列表
    """
    elo_map = dict(DEFAULT_ELO_MAP)
    if elo_overrides:
        elo_map.update(elo_overrides)

    references = []
    for name in anchor_names:
        elo = elo_map.get(name, 1200.0)  # 未知 agent 默认 1200
        references.append(
            ReferenceAgent(
                name=name,
                elo=elo,
                agent_factory=lambda pid, n=name: AgentLoader.load(n, pid),
                is_anchor=True,
            )
        )
    return references


def _flat_battle_worker(
    ga_bytes: bytes,
    ref_bytes: bytes,
    n_battles: int,
    max_rounds: int,
) -> List[Dict[str, Any]]:
    """在子进程中执行 (个体, 参照物) 的所有对战。

    一次创建网络，执行 n_battles * 2 局对战（先后手各半），
    避免重复序列化/反序列化的开销。
    """
    _reconfigure_loguru()
    ga_agent = AgentLoader.deserialize(ga_bytes)
    ref_agent = AgentLoader.deserialize(ref_bytes)
    ga_agent = _ensure_compatible(ga_agent)
    ref_agent = _ensure_compatible(ref_agent)

    results = []
    for i in range(n_battles * 2):
        first_player = i % 2  # 0: ga 先手, 1: ga 后手
        result = _execute_battle(ga_agent, ref_agent, first_player, max_rounds)
        result["_individual_id"] = ga_agent.name if hasattr(ga_agent, "name") else None
        result["_ref_name"] = ref_agent.name if hasattr(ref_agent, "name") else None
        results.append(result)
    return results


class BattleSelection:
    """对战选拔 —— 编排个体与参照物的对战、ELO 评分、种子选拔"""

    def __init__(
        self,
        config: GABattleConfig,
        device: str = "cuda",
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
        num_seeds: int = 8,
        elo_k_factor: float = 64.0,
        elo_initial: float = 1200.0,
        anchor_references: Optional[List[ReferenceAgent]] = None,
    ):
        self.config = config
        self.device = device
        self.hidden_dim = hidden_dim
        self.enable_auxiliary = enable_auxiliary
        self.num_seeds = num_seeds
        self.elo = ELORating(initial_rating=elo_initial, k_factor=elo_k_factor)
        self._simulator = BattleSimulator(config)
        self.anchor_references = anchor_references or DEFAULT_ANCHOR_REFERENCES

    def evaluate_population(
        self,
        population: Population,
        dynamic_references: Optional[List[ReferenceAgent]] = None,
    ) -> List[Individual]:
        """评估种群中所有个体的 ELO 评分（扁平化并行）。

        将所有 (个体, 参照物) 对战一次性提交到 ProcessPoolExecutor，
        消除外层串行循环，充分利用多核。
        """
        self.elo.reset_individuals()

        # 构建参照物列表
        references = list(self.anchor_references)
        if dynamic_references:
            references.extend(dynamic_references)

        for ref in references:
            self.elo.ensure_player(ref.name)
            if ref.is_anchor:
                # 固定参照物：强制使用硬编码 ELO，不参与更新
                self.elo._ratings[ref.name] = ref.elo
            else:
                # 动态参照物：用上一轮实际 ELO
                self.elo._ratings[ref.name] = self.elo.get_dynamic_ref_rating(ref.name)

        # ── 步骤 1：预序列化所有参照物 agent ──
        ref_bytes: Dict[str, bytes] = {}
        for ref in references:
            agent = ref.agent_factory(1)
            ref_bytes[ref.name] = AgentLoader.serialize(agent)

        # ── 步骤 2：构建任务列表 ──
        tasks: List[dict] = []
        n_battles = self.config.n_battles
        max_rounds = self.config.max_rounds

        for individual in population.individuals:
            self.elo.ensure_player(individual.id)
            ga_agent = self._create_ga_agent(individual)
            ga_bytes = AgentLoader.serialize(ga_agent)

            for ref in references:
                tasks.append({
                    "individual_id": individual.id,
                    "ga_bytes": ga_bytes,
                    "ref_name": ref.name,
                    "ref_bytes": ref_bytes[ref.name],
                })

        # ── 步骤 3：一次性并行提交 ──
        max_workers = min(len(tasks), self.config.max_workers)
        task_timeout = n_battles * 2 * _BATTLE_TIMEOUT

        logger.info(
            f"Selection: {len(tasks)} tasks (pop={len(population.individuals)} x refs={len(references)} x "
            f"n_battles={n_battles}), max_workers={max_workers}"
        )

        spawn_ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=spawn_ctx) as executor:
            futures = {}
            for task in tasks:
                future = executor.submit(
                    _flat_battle_worker,
                    task["ga_bytes"],
                    task["ref_bytes"],
                    n_battles,
                    max_rounds,
                )
                futures[future] = task

            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    battle_results = future.result(timeout=task_timeout)
                except Exception as e:
                    logger.error(f"Battle task failed: {task['individual_id']} vs {task['ref_name']}: {e}")
                    self.elo.update(task["individual_id"], task["ref_name"], 0.5)
                    completed += 1
                    continue

                for result in battle_results:
                    if "error" in result:
                        self.elo.update(task["individual_id"], task["ref_name"], 0.5)
                        logger.debug(
                            f"Battle ERROR: {task['individual_id']} vs {task['ref_name']}: {result.get('error', 'unknown')}"
                        )
                        continue

                    game_result = result.get("result")
                    if game_result == "agent1_win":
                        self.elo.update(task["individual_id"], task["ref_name"], 1.0)
                    elif game_result == "agent2_win":
                        self.elo.update(task["individual_id"], task["ref_name"], 0.0)
                    else:
                        self.elo.update(task["individual_id"], task["ref_name"], 0.5)

                completed += 1
                logger.info(f"Selection progress: {completed}/{len(tasks)} tasks completed")

        # ── 步骤 4：回写 ELO 评分并排序 ──
        for individual in population.individuals:
            individual.elo_rating = self.elo.get_rating(individual.id)

        # 保存动态参照物当轮 ELO（供下一代使用）
        for ref in references:
            if not ref.is_anchor:
                self.elo.save_dynamic_ref_rating(ref.name, self.elo.get_rating(ref.name))

        ratings = [ind.elo_rating for ind in population.individuals]
        logger.info(
            f"Selection done: best_elo={max(ratings):.1f}, worst_elo={min(ratings):.1f}, "
            f"mean_elo={sum(ratings)/len(ratings):.1f}"
        )

        return sorted(population.individuals, key=lambda ind: ind.elo_rating, reverse=True)

    def select_seeds(self, ranked_individuals: List[Individual]) -> List[Individual]:
        """从排名列表中选拔种子。

        Args:
            ranked_individuals: 按 ELO 降序排列的个体列表

        Returns:
            种子列表（前 num_seeds 个）
        """
        seeds = ranked_individuals[:self.num_seeds]
        for i, seed in enumerate(seeds):
            seed.seed_rank = i + 1
        logger.info(f"Seeds selected: {[s.id for s in seeds]} (top {len(seeds)})")
        return seeds

    def select_seeds_diverse(
        self,
        ranked_individuals: List[Individual],
        hard_threshold: float = 0.95,
        soft_threshold: float = 0.85,
        soft_max_per_family: int = 2,
    ) -> List[Individual]:
        """基于 ELO 排名进行多样性感知的种子选拔。

        在按 ELO 降序遍历时：
        1. 硬去重：cos > hard_threshold → 视为相同模型，跳过
        2. 软限流：已选同类 ≥ soft_max_per_family 时跳过
        3. 兜底：不足 n_seeds 时逐级放宽限制

        Args:
            ranked_individuals: 按 ELO 降序排列的个体列表
            hard_threshold: 硬去重的余弦相似度阈值（默认 0.95）
            soft_threshold: 同一家族的余弦阈值（默认 0.85）
            soft_max_per_family: 每家族最大名额（默认 2）

        Returns:
            去重后的种子列表（前 num_seeds 个）
        """
        import numpy as np

        # 同时存储 Individual 和其 weights，避免后续通过 index() 回查
        selected: List[Individual] = []
        selected_weights: List[np.ndarray] = []

        for candidate in ranked_individuals:
            w_candidate = candidate.genome.weights

            # 检查是否与已选种子硬重复
            is_hard_dup = False
            family_count = 0
            for sel_w in selected_weights:
                cos = cosine_similarity(w_candidate, sel_w)

                if cos > hard_threshold:
                    is_hard_dup = True
                    logger.debug(
                        f"Hard duplicate: {candidate.id} (ELO={candidate.elo_rating:.0f}) "
                        f"cos={cos:.4f}"
                    )
                    break

                if cos > soft_threshold:
                    family_count += 1

            if is_hard_dup:
                continue

            if family_count >= soft_max_per_family:
                logger.debug(
                    f"Soft throttled: {candidate.id} (ELO={candidate.elo_rating:.0f}) "
                    f"family already has {family_count} seeds"
                )
                continue

            # 通过筛选
            selected.append(candidate)
            selected_weights.append(w_candidate)
            if len(selected) >= self.num_seeds:
                break

        # ── 兜底：不足 num_seeds，逐级放宽 ──
        if len(selected) < self.num_seeds:
            selected_ids = {ind.id for ind in selected}
            remaining = [ind for ind in ranked_individuals if ind.id not in selected_ids]

            for fallback_level, (st, sm) in enumerate([
                (soft_threshold + 0.05, 3),   # Level 1
                (soft_threshold + 0.05, 4),   # Level 2
                (1.0, self.num_seeds),         # Level 3: 完全降级
            ]):
                for candidate in remaining:
                    if candidate.id in selected_ids:
                        continue
                    if len(selected) >= self.num_seeds:
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

                if len(selected) >= self.num_seeds:
                    logger.warning(
                        f"Seed dedup fallback level {fallback_level}: "
                        f"got {len(selected)} seeds"
                    )
                    break

        # 最终兜底：仍然不够，直接按 ELO 取
        if len(selected) < self.num_seeds:
            logger.error(
                f"Seed dedup exhausted: only {len(selected)}/{self.num_seeds} seeds selected. "
                f"Using ELO-only fallback."
            )
            selected = ranked_individuals[:self.num_seeds]

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

        return selected[:self.num_seeds]

    def _create_ga_agent(self, individual: Individual) -> GAWarAgent:
        """从个体基因组创建 GAWarAgent（用于对战）。

        Args:
            individual: 个体

        Returns:
            GAWarAgent 实例
        """
        import torch
        device = torch.device(self.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.hidden_dim,
            enable_auxiliary=self.enable_auxiliary,
        ).to(device)
        load_genome_to_network(network, individual.genome)
        network.eval()
        return GAWarAgent(
            player_id=0,
            network=network,
            device=device,
        )
