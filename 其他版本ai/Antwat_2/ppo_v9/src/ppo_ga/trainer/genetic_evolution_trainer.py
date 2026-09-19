from typing import Dict, Any, Optional, List
import time
import os

import torch
from loguru import logger

from ..config.path_config import PathConfig
from ..evolution.population_manager import PopulationManager, Population, Individual
from ..evolution.crossover import layer_wise_crossover
from ..evolution.mutation import gaussian_mutation
from ..selection.battle_selection import BattleSelection, ReferenceAgent, build_anchor_references
from ..evaluation.baseline_evaluator import BaselineEvaluator
from ..league.opponent_pool import OpponentPool
from ..league.battle_shared_payoff import BattleSharedPayoff
from ..league.opponent_selector import OpponentSelector
from ..genome.genome import load_genome_to_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..battle.opponent_agent import OpponentAgent
from ..monitor.system_metrics_sampler import SystemMetricsSampler
from .checkpoint_manager import GACheckpointManager
from .logging_subsystem import GALoggingSubsystem


class GeneticEvolutionTrainer:
    """遗传进化训练器 —— 编排整个进化流程"""

    def __init__(self, config: Dict[str, Any], path_config: PathConfig, run_id: str):
        self.config = config
        self._path_config = path_config
        self._run_id = run_id
        self.device = torch.device(config["system"]["device"])

        # ── 路径（通过 PathConfig 管理，与 ppo_v2 对齐）──
        self._run_dir = str(path_config.get_run_dir(run_id))
        self._checkpoint_dir = str(path_config.get_checkpoint_dir(run_id))
        self._league_dir = str(path_config.get_league_dir(run_id))
        self._generations_dir = str(path_config.get_generations_dir(run_id))
        # 将 _output_dir 写入 config，保持向后兼容
        config["_output_dir"] = self._run_dir

        # ── 核心组件 ──
        self.population_mgr = PopulationManager(
            population_size=config["evolution"]["population_size"],
            elitism_count=config["evolution"]["elitism_count"],
            hidden_dim=config["network"]["hidden_dim"],
            enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
        )

        # 构建锚定参照物列表
        anchor_names = config["selection"].get("anchor_references", ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"])
        anchor_refs = build_anchor_references(anchor_names)

        self.battle_selection = BattleSelection(
            config=self._create_battle_config(
                n_battles=config["selection"].get("n_battles", 1)
            ),
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
            enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
            num_seeds=config["evolution"]["num_seeds"],
            elo_k_factor=config["selection"]["elo_k_factor"],
            elo_initial=config["selection"]["elo_initial"],
            anchor_references=anchor_refs,
        )

        self.baseline_evaluator = BaselineEvaluator(
            config=self._create_battle_config(),
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
            enable_auxiliary=config.get("network", {}).get("enable_auxiliary", True),
            enabled=config["baseline_battle"]["enabled"],
            interval=config["baseline_battle"]["interval"],
            n_battles=config["baseline_battle"]["n_battles"],
            baseline_agents=config["baseline_battle"].get("agents", None),
        )

        # ── 对手池 ──
        self.payoff = BattleSharedPayoff(
            decay=config["opponent_pool"].get("decay", 0.99),
            min_win_rate_games=config["opponent_pool"].get("min_win_rate_games", 8),
        )
        self.opponent_pool = OpponentPool(
            max_size=config["opponent_pool"]["max_size"],
            min_games_threshold=config["opponent_pool"]["min_games_threshold"],
            payoff=self.payoff,
        )
        self.opponent_selector = OpponentSelector(
            payoff=self.payoff,
            opponent_pool=self.opponent_pool,
            exploit_prob=config["opponent_pool"]["exploit_prob"],
        )

        # ── 日志与检查点 ──
        self.logging = GALoggingSubsystem(config, path_config, run_id)
        self.checkpoint_mgr = GACheckpointManager(self.device)

        # ── 系统监控（与 ppo_v2 对齐）──
        self._system_sampler = SystemMetricsSampler(
            output_dir=self.logging.system_dir,
            sample_interval=10,
        )

        # ── 训练状态 ──
        self._generation = 0
        self._best_elo_history: List[float] = []
        self._start_time = time.time()

        logger.info(
            f"Training initialized: pop_size={config['evolution']['population_size']}, "
            f"num_seeds={config['evolution']['num_seeds']}, "
            f"device={config['system']['device']}, "
            f"max_generations={config['evolution']['max_generations']}"
        )

    def train(self) -> None:
        """主训练循环"""
        max_generations = self.config["evolution"]["max_generations"]
        convergence_window = self.config["evolution"]["convergence_window"]
        convergence_threshold = self.config["evolution"]["convergence_threshold"]

        # 记录训练启动快照
        self.logging.log_training_start()
        logger.info(f"GA Training started: max_generations={max_generations}")

        # 启动系统监控
        self._system_sampler.start()

        try:
            for gen in range(1, max_generations + 1):
                self._generation = gen
                gen_start_time = time.time()

                # ── 阶段一：种群生成 ──
                self._system_sampler.set_phase("evolution")
                logger.info(f"Gen {gen} - Phase: population")
                if gen == 1:
                    population = self.population_mgr.init_population()
                    # 初代种群创建后立即落盘，便于离线分析（不影响后续保存逻辑）
                    self._save_checkpoint(gen, population)
                else:
                    population = self.population_mgr.evolve(
                        seeds=self._last_seeds,
                        generation=gen,
                        crossover_fn=layer_wise_crossover,
                        mutation_fn=gaussian_mutation,
                        current_mutation_scale=self._current_mutation_scale(gen),
                    )

                # ── 阶段二：对战评估 ──
                self._system_sampler.set_phase("battle")
                logger.info(f"Gen {gen} - Phase: battle")
                dynamic_refs = self._build_dynamic_references(gen)
                ranked = self.battle_selection.evaluate_population(population, dynamic_refs)

                # ── 阶段三：种子选拔 ──
                logger.info(f"Gen {gen} - Phase: selection")
                dedup_cfg = self.config.get("elite_dedup", {})
                if dedup_cfg.get("enabled", True):
                    seeds = self.battle_selection.select_seeds_diverse(
                        ranked,
                        hard_threshold=dedup_cfg.get("hard_threshold", 0.95),
                        soft_threshold=dedup_cfg.get("soft_threshold", 0.85),
                        soft_max_per_family=dedup_cfg.get("soft_max_per_family", 2),
                    )
                else:
                    seeds = self.battle_selection.select_seeds(ranked)
                self._last_seeds = seeds

                # 更新种群统计
                population.best_elo = ranked[0].elo_rating if ranked else 0.0
                population.mean_elo = sum(ind.elo_rating for ind in ranked) / max(len(ranked), 1)
                population.diversity = PopulationManager.compute_diversity(population)

                # ── 阶段四：基线验证 ──
                logger.info(f"Gen {gen} - Phase: baseline")
                baseline_results = self.baseline_evaluator.evaluate_seeds(seeds, gen)

                # ── 阶段五：对手池更新（仅精英加入）──
                logger.info(f"Gen {gen} - Phase: opponent_pool")
                elitism_count = self.config["evolution"]["elitism_count"]
                for seed in seeds[:elitism_count]:
                    checkpoint_path = self._save_seed_model(seed, gen)
                    self.opponent_pool.add(seed.id, checkpoint_path, gen)

                # ── 阶段六：日志与持久化 ──
                self._system_sampler.set_phase("logging")
                logger.info(f"Gen {gen} - Phase: logging")
                gen_duration = time.time() - gen_start_time
                self._best_elo_history.append(population.best_elo)

                self.logging.log_generation(
                    generation=gen,
                    population=population,
                    seeds=seeds,
                    baseline_results=baseline_results,
                    duration=gen_duration,
                )

                # 定期保存 checkpoint
                save_interval = self.config["logging"]["save_interval"]
                if gen % save_interval == 0:
                    self._save_checkpoint(gen, population)

                # 持久化对手池
                self._save_league_state()

                # CUDA 缓存清理（防止显存碎片化）
                if self.device.type == "cuda" and gen % 3 == 0:
                    torch.cuda.empty_cache()

                # ── 收敛判断 ──
                if gen >= convergence_window:
                    recent = self._best_elo_history[-convergence_window:]
                    if max(recent) - min(recent) < convergence_threshold:
                        logger.info(f"Converged at generation {gen}")
                        break
        finally:
            # 停止系统监控
            self._system_sampler.stop()
            logger.info("System metrics sampler stopped.")

        # 训练完成
        total_time = time.time() - self._start_time
        self.logging.log_training_complete(self._generation, total_time)
        self.logging.close()

    def _current_mutation_scale(self, generation: int) -> float:
        """计算当前变异幅度（含衰减）"""
        base_scale = self.config["mutation"]["scale"]
        decay = self.config["mutation"]["decay"]
        return base_scale * (decay ** (generation - 1))

    def _build_dynamic_references(self, generation: int) -> List[ReferenceAgent]:
        """构建动态参照物列表"""
        dynamic_refs = []

        # 从对手池选动态参照物
        n_dynamic = self.config.get("selection", {}).get("dynamic_references", 0)
        for _ in range(n_dynamic):
            opp_id = self.opponent_selector.select()
            if opp_id:
                checkpoint_path = self.opponent_pool.get_checkpoint_path(opp_id)
                if checkpoint_path and os.path.exists(checkpoint_path):
                    ref = self._create_reference_from_checkpoint(
                        opp_id, checkpoint_path
                    )
                    if ref:
                        dynamic_refs.append(ref)

        # 上一代冠军（如果有）
        if self.config.get("selection", {}).get("include_previous_champion", True):
            if hasattr(self, '_last_seeds') and self._last_seeds:
                champion = self._last_seeds[0]
                ref = ReferenceAgent(
                    name=f"champion_gen_{generation-1}",
                    elo=champion.elo_rating,
                    agent_factory=lambda pid, g=champion: self._create_agent_from_individual(g, pid),
                    is_anchor=False,
                )
                dynamic_refs.append(ref)

        if dynamic_refs:
            logger.info(f"Dynamic references: {len(dynamic_refs)} opponents selected")

        return dynamic_refs

    def _create_reference_from_checkpoint(
        self, opponent_id: str, checkpoint_path: str
    ) -> Optional[ReferenceAgent]:
        """从对手池 checkpoint 创建参照物"""
        try:
            def factory(pid, cp=checkpoint_path):
                agent = OpponentAgent.from_checkpoint(
                    checkpoint_path=cp,
                    device=self.device,
                    hidden_dim=self.config["network"]["hidden_dim"],
                )
                from ..battle.ppo_war_agent import PPOWarAgent
                from ..env.observation import ObservationEncoder
                from ..env.action_mask import ActionMaskHandler

                return PPOWarAgent(
                    player_id=pid,
                    ppo_agent=agent,
                    observation_encoder=ObservationEncoder(),
                    action_mask_handler=ActionMaskHandler(),
                )

            elo = self.battle_selection.elo.get_dynamic_ref_rating(opponent_id)

            return ReferenceAgent(
                name=opponent_id,
                elo=elo,
                agent_factory=factory,
                is_anchor=False,
            )
        except Exception as e:
            logger.warning(f"Failed to create reference from {checkpoint_path}: {e}")
            return None

    def _create_agent_from_individual(self, individual: Individual, player_id: int):
        """从 Individual 创建对战 Agent"""
        from ..battle.ga_war_agent import GAWarAgent

        network = AntWarPolicyValueNetwork(
            hidden_dim=self.config["network"]["hidden_dim"],
            enable_auxiliary=self.config.get("network", {}).get("enable_auxiliary", True),
        ).to(self.device)
        load_genome_to_network(network, individual.genome)
        network.eval()
        return GAWarAgent(player_id=player_id, network=network, device=self.device)

    def _save_seed_model(self, seed: Individual, generation: int) -> str:
        """保存种子模型到文件"""
        seed_dir = os.path.join(self._generations_dir, f"gen_{generation:04d}", "seed_models")
        os.makedirs(seed_dir, exist_ok=True)

        filepath = os.path.join(seed_dir, f"seed_{seed.seed_rank}.pt")

        network = AntWarPolicyValueNetwork(
            hidden_dim=self.config["network"]["hidden_dim"],
            enable_auxiliary=self.config.get("network", {}).get("enable_auxiliary", True),
        ).to(self.device)
        load_genome_to_network(network, seed.genome)

        torch.save({
            "generation": generation,
            "individual_id": seed.id,
            "seed_rank": seed.seed_rank,
            "elo_rating": seed.elo_rating,
            "policy_state_dict": network.state_dict(),
            "parent_ids": seed.parent_ids,
        }, filepath)

        logger.info(f"Seed model saved: {seed.id} (elo={seed.elo_rating:.1f}) -> {filepath}")
        return filepath

    def _save_checkpoint(self, generation: int, population: Population) -> None:
        """保存完整训练状态"""
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        filepath = os.path.join(self._checkpoint_dir, f"checkpoint_gen_{generation:04d}.pt")
        self.checkpoint_mgr.save(
            filepath=filepath,
            generation=generation,
            population=population,
            opponent_pool=self.opponent_pool,
            payoff=self.payoff,
            config=self.config,
        )

    def _save_league_state(self) -> None:
        """持久化对手池状态"""
        os.makedirs(self._league_dir, exist_ok=True)

        self.opponent_pool.save_state(os.path.join(self._league_dir, "pool.json"))
        self.payoff.save_state(os.path.join(self._league_dir, "payoff.json"))
        logger.info(f"League state saved (pool_size={len(self.opponent_pool._opponents)})")

    def _create_battle_config(self, n_battles: int | None = None):
        """创建对战配置"""
        from ..battle.battle_config import GABattleConfig
        if n_battles is None:
            n_battles = self.config["baseline_battle"]["n_battles"]
        return GABattleConfig(
            max_rounds=self.config["env"]["max_steps"],
            n_battles=n_battles,
            max_workers=self.config["baseline_battle"].get("max_workers", 12),
            device=self.config["system"]["device"],
        )
