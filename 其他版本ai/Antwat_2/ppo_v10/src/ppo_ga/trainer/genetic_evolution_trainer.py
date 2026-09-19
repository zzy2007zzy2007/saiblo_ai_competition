from typing import Dict, Any, List
import time
import os

import torch
from loguru import logger

from ..config.path_config import PathConfig
from ..evolution.population_manager import PopulationManager, Population, Individual
from ..evolution.crossover import layer_wise_crossover
from ..evolution.mutation import gaussian_mutation
from ..selection.battle_selection import BattleSelection
from ..selection.touchstone_selector import TouchstoneConfig
from ..selection.round_robin import RoundRobinConfig
from ..selection.novice_pruner import NovicePruner
from ..league.elite_seed_pool import EliteSeedPool
from ..league.battle_shared_payoff import BattleSharedPayoff
from ..evaluation.baseline_evaluator import BaselineEvaluator
from ..genome.genome import load_genome_to_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..monitor.system_metrics_sampler import SystemMetricsSampler
from .checkpoint_manager import GACheckpointManager
from .logging_subsystem import GALoggingSubsystem


class GeneticEvolutionTrainer:
    """遗传进化训练器 —— 编排整个进化流程（v10 双池双阶段）"""

    def __init__(self, config: Dict[str, Any], path_config: PathConfig, run_id: str):
        self.config = config
        self._path_config = path_config
        self._run_id = run_id
        self.device = torch.device(config["system"]["device"])

        # ── 路径（通过 PathConfig 管理）──
        self._run_dir = str(path_config.get_run_dir(run_id))
        self._checkpoint_dir = str(path_config.get_checkpoint_dir(run_id))
        self._league_dir = str(path_config.get_league_dir(run_id))
        self._generations_dir = str(path_config.get_generations_dir(run_id))
        config["_output_dir"] = self._run_dir

        # ── 核心组件 ──
        self.population_mgr = PopulationManager(
            population_size=config["evolution"]["population_size"],
            elitism_count=config["evolution"]["elitism_count"],
            hidden_dim=config["network"]["hidden_dim"],
        )

        # Gen 1 专用种群规模（None 表示与 population_size 一致）
        _g1 = config["evolution"].get("gen1_population_size")
        self._gen1_population_size: int | None = int(_g1) if _g1 is not None else None

        # ── 精英种子池 ──
        self.elite_seed_pool = EliteSeedPool()

        # ── 共享 payoff ──
        self.payoff = BattleSharedPayoff(
            decay=0.99,
            min_win_rate_games=8,
        )

        # ── 选拔模块（基准筛选：touchstone 二分种子；精排：循环赛）──
        orig_cfg = TouchstoneConfig(
            M=config["touchstone"]["M"],
            max_seeds=config["touchstone"]["max_seeds"],
            n_battle=config["touchstone"]["n_battle"],
            max_candidate_fails=config["touchstone"]["max_candidate_fails"],
            max_workers=config["touchstone"]["max_workers"],
            max_rounds=config["env"]["max_steps"],
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
            candidate_min_rank=config["touchstone"].get("candidate_min_rank", 30),
            candidate_max_rank=config["touchstone"].get("candidate_max_rank", 15),
            next_candidate_rank=config["touchstone"].get("next_candidate_rank", 20),
        )
        rr_cfg = RoundRobinConfig(
            top_n=config["round_robin"]["top_n"],
            n_battles=config["round_robin"]["n_battles"],
            max_workers=config["round_robin"]["max_workers"],
            max_rounds=config["env"]["max_steps"],
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
        )

        # ── NoviceAI 剪枝器（可选）──
        pruning_config = config.get("pruning", {})
        pruner = None
        if pruning_config.get("enabled", False):
            pruner = NovicePruner(
                config=pruning_config,
                device=config["system"]["device"],
                hidden_dim=config["network"]["hidden_dim"],
                max_rounds=config["env"]["max_steps"],
            )
            logger.info(
                f"NovicePruner enabled: "
                f"early(gen1-{pruner.early_gens})={pruning_config['early']}, "
                f"late(gen{pruner.early_gens+1}+)={pruning_config['late']}"
            )

        self.battle_selection = BattleSelection(
            touchstone_config=orig_cfg,
            round_robin_config=rr_cfg,
            payoff=self.payoff,
            dedup_config=config.get("elite_dedup", {}),
            num_seeds=config["evolution"]["num_seeds"],
            elitism_count=config["evolution"]["elitism_count"],
            pruner=pruner,
        )

        # ── 基线评估器（诊断环节，不变）──
        self.baseline_evaluator = BaselineEvaluator(
            config=self._create_battle_config(),
            device=config["system"]["device"],
            hidden_dim=config["network"]["hidden_dim"],
            enabled=config["baseline_battle"]["enabled"],
            interval=config["baseline_battle"]["interval"],
            n_battles=config["baseline_battle"]["n_battles"],
            baseline_agents=config["baseline_battle"].get("agents", None),
        )

        # ── 日志与检查点 ──
        self.logging = GALoggingSubsystem(config, path_config, run_id)
        self.checkpoint_mgr = GACheckpointManager(self.device)

        # ── 系统监控 ──
        self._system_sampler = SystemMetricsSampler(
            output_dir=self.logging.system_dir,
            sample_interval=10,
        )

        # ── 训练状态 ──
        self._generation = 0
        self._last_seeds: List[Individual] = []
        self._best_rating_history: List[float] = []
        self._start_time = time.time()

        logger.info(
            f"Training initialized: pop_size={config['evolution']['population_size']}, "
            f"num_seeds={config['evolution']['num_seeds']}, "
            f"device={config['system']['device']}, "
            f"max_generations={config['evolution']['max_generations']}"
        )

    def train(self) -> None:
        """主训练循环（v10 双池双阶段流程）"""
        max_generations = self.config["evolution"]["max_generations"]
        convergence_window = self.config["evolution"]["convergence_window"]
        convergence_threshold = self.config["evolution"]["convergence_threshold"]

        # 记录训练启动快照
        self.logging.log_training_start()
        logger.info(f"GA Training started (v10): max_generations={max_generations}")

        # 启动系统监控
        self._system_sampler.start()

        try:
            for gen in range(1, max_generations + 1):
                self._generation = gen
                gen_start_time = time.time()

                # ── 阶段 1：种群生成（不变）──
                self._system_sampler.set_phase("evolution")
                logger.info(f"Gen {gen} - Phase: population")
                if gen == 1 or not self._last_seeds:
                    population = self.population_mgr.init_population(
                        generation=gen,
                        size=self._gen1_population_size if gen == 1 else None,
                    )
                else:
                    population = self.population_mgr.evolve(
                        seeds=self._last_seeds,
                        generation=gen,
                        crossover_fn=layer_wise_crossover,
                        mutation_fn=gaussian_mutation,
                        current_mutation_scale=self._current_mutation_scale(gen),
                    )

                # ── 阶段 2：个体选择（两阶段：基准筛选(二分种子) + 循环赛）──
                self._system_sampler.set_phase("battle")
                logger.info(f"[phase=battle][gen={gen}] start: stages=pruning,touchstone,round_robin")
                gen_dir = os.path.join(self._generations_dir, f"gen_{gen:04d}")
                ranked_all, seeds, elites = self.battle_selection.evaluate_and_select(
                    population, generation=gen, gen_dir=gen_dir
                )
                self._last_seeds = seeds

                # 剪枝后可能无存活个体；仍继续记录与持久化本代状态
                if not seeds:
                    logger.error(
                        f"Gen {gen}: No seeds produced — "
                        f"all individuals were pruned or selection failed. "
                        f"Next generation will re-initialize population."
                    )
                    self._last_seeds = []

                # 在 RR 后 trueskill 可能更新，重新按当前 mu 排序
                ranked_all = sorted(
                    ranked_all, key=lambda ind: ind.trueskill_mu, reverse=True
                )

                # 更新种群统计
                population.best_rating = ranked_all[0].trueskill_mu if ranked_all else 0.0
                mu_values = [ind.trueskill_mu for ind in ranked_all]
                population.mean_rating = sum(mu_values) / max(len(mu_values), 1)
                population.diversity = PopulationManager.compute_diversity(population)

                # ── 阶段 3：精英种子池更新 ──
                seed_dir = os.path.join(self._generations_dir, f"gen_{gen:04d}", "seed_models")
                self.elite_seed_pool.add_seeds(seeds, elites, gen, seed_dir)

                # ── 阶段 4：诊断环节（基线验证，不变）──
                logger.info(f"[phase=baseline_battle][gen={gen}] prepare")
                baseline_results = self.baseline_evaluator.evaluate_seeds(seeds, gen, gen_dir=gen_dir)

                # ── 阶段 5：日志与持久化 ──
                self._system_sampler.set_phase("logging")
                logger.info(f"Gen {gen} - Phase: logging")
                gen_duration = time.time() - gen_start_time
                self._best_rating_history.append(population.best_rating)

                self.logging.log_generation(
                    generation=gen,
                    population=population,
                    seeds=seeds,
                    baseline_results=baseline_results,
                    duration=gen_duration,
                    num_benchmark_agents_used=self.battle_selection.benchmark.unique_candidates,
                    benchmark_kendall_tau_final=None,
                    illegal_stats={
                        "touchstone_illegal": self.battle_selection.benchmark.illegal_stats.get("total_illegal", 0),
                        "round_robin_illegal": self.battle_selection.round_robin.illegal_stats.get("total_illegal", 0),
                        "baseline_illegal": sum(
                            agg.get("total_illegal", 0)
                            for seed_results in baseline_results.values()
                            if isinstance(seed_results, dict)
                            for agg in seed_results.values()
                            if isinstance(agg, dict)
                        ),
                        "avg_rounds": self.battle_selection.round_robin.illegal_stats.get("avg_rounds", 0),
                        "touchstone_avg_rounds": self.battle_selection.benchmark.illegal_stats.get("avg_rounds", 0),
                        "round_robin_avg_rounds": self.battle_selection.round_robin.illegal_stats.get("avg_rounds", 0),
                        "baseline_avg_rounds": round(
                            sum(
                                agg.get("avg_rounds", 0)
                                for seed_results in baseline_results.values()
                                if isinstance(seed_results, dict)
                                for agg in seed_results.values()
                                if isinstance(agg, dict)
                            ) / max(
                                sum(
                                    1
                                    for seed_results in baseline_results.values()
                                    if isinstance(seed_results, dict)
                                    for agg in seed_results.values()
                                    if isinstance(agg, dict)
                                ), 1
                            ), 1
                        ),
                    },
                )

                # 保存 seed_models
                for seed in seeds:
                    self._save_seed_model(seed, gen)

                # 定期保存 checkpoint
                save_interval = self.config["logging"]["save_interval"]
                if gen % save_interval == 0:
                    self._save_checkpoint(gen, population)

                # 持久化 league 状态
                self._save_league_state()

                # CUDA 缓存清理
                if self.device.type == "cuda" and gen % 3 == 0:
                    torch.cuda.empty_cache()

                # ── 收敛判断 ──
                if gen >= convergence_window:
                    recent = self._best_rating_history[-convergence_window:]
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

    def _save_seed_model(self, seed: Individual, generation: int) -> str:
        """保存种子模型到文件"""
        seed_dir = os.path.join(self._generations_dir, f"gen_{generation:04d}", "seed_models")
        os.makedirs(seed_dir, exist_ok=True)

        filepath = os.path.join(seed_dir, f"seed_{seed.seed_rank}.pt")

        network = AntWarPolicyValueNetwork(
            hidden_dim=self.config["network"]["hidden_dim"],
        ).to(self.device)
        load_genome_to_network(network, seed.genome)

        torch.save({
            "generation": generation,
            "individual_id": seed.id,
            "seed_rank": seed.seed_rank,
            "trueskill_mu": seed.trueskill_mu,
            "trueskill_sigma": seed.trueskill_sigma,
            "policy_state_dict": network.state_dict(),
            "parent_ids": seed.parent_ids,
        }, filepath)

        logger.info(f"Seed model saved: {seed.id} (mu={seed.trueskill_mu:.2f}) -> {filepath}")
        return filepath

    def _save_checkpoint(self, generation: int, population: Population) -> None:
        """保存完整训练状态"""
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        filepath = os.path.join(self._checkpoint_dir, f"checkpoint_gen_{generation:04d}.pt")
        self.checkpoint_mgr.save(
            filepath=filepath,
            generation=generation,
            population=population,
            payoff=self.payoff,
            config=self.config,
        )

    def _save_league_state(self) -> None:
        """持久化 league 状态（payoff + elite_seed_pool）"""
        os.makedirs(self._league_dir, exist_ok=True)
        self.elite_seed_pool.save_state(os.path.join(self._league_dir, "elite_seed_pool.json"))
        self.payoff.save_state(os.path.join(self._league_dir, "payoff.json"))
        logger.info(
            f"League state saved "
            f"(elite_seed_pool={len(self.elite_seed_pool.entries)})"
        )

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
