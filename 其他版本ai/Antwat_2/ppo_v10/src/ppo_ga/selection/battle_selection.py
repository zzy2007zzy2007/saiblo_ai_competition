"""
PPO_V10 选拔模块 — 编排剪枝 + 基准筛选（二分种子）+ 循环赛三阶段流程。

v10 改造（touchstone 接入）：
  • 阶段 0（可选）：NovicePruner 剪枝
    （仅与 NoviceAI 对战，全胜者进入下一阶段）
  • 阶段 1：TouchstoneSelector 替换 BenchmarkTest
    （种群内部互打 → 二分种子 → 全员 mu 排名）
  • 阶段 2：RoundRobinTournament（取 TopN → seeds）保持不变
"""

from typing import Optional
from loguru import logger

from ..evolution.population_manager import Population
from ..league.battle_shared_payoff import BattleSharedPayoff
from .touchstone_selector import TouchstoneSelector, TouchstoneConfig
from .round_robin import RoundRobinTournament, RoundRobinConfig
from .novice_pruner import NovicePruner


class BattleSelection:
    """对战选拔 — 编排剪枝 + 基准筛选（二分种子） + 循环赛三阶段"""

    def __init__(
        self,
        touchstone_config: TouchstoneConfig,
        round_robin_config: RoundRobinConfig,
        payoff: BattleSharedPayoff,
        dedup_config: dict,
        num_seeds: int = 8,
        elitism_count: int = 2,
        pruner: Optional[NovicePruner] = None,
    ):
        self._pruner = pruner
        self._benchmark = TouchstoneSelector(touchstone_config, payoff)
        self._round_robin = RoundRobinTournament(round_robin_config, payoff)
        self._payoff = payoff
        self._dedup_config = dedup_config
        self.num_seeds = num_seeds
        self.elitism_count = elitism_count

    def evaluate_and_select(
        self,
        population: Population,
        generation: int = 1,
        gen_dir: str = None,
    ) -> tuple:
        """
        执行完整的三阶段选拔。

        Returns:
            (ranked_all, seeds, elites)
            - ranked_all: 选拔后全部个体的 TrueSkill 排名
            - seeds: 循环赛选出的种子列表
            - elites: 前 elitism_count 个种子
        """
        # ── 阶段 0：NoviceAI 剪枝（可选）──
        if self._pruner is not None:
            survivors, prune_stats = self._pruner.prune(
                population.individuals, generation=generation, gen_dir=gen_dir
            )
            population.individuals = survivors

            if len(survivors) == 0:
                logger.warning(
                    f"[phase=pruning][gen={generation}] skip_next: "
                    "reason=all_pruned, skip=touchstone,round_robin"
                )
                population.best_rating = 0.0
                population.mean_rating = 0.0
                return [], [], []

            if self._should_skip_touchstone(len(survivors)):
                logger.warning(
                    f"[phase=touchstone][gen={generation}] skip: "
                    f"reason=insufficient_survivors, survivors={len(survivors)}, "
                    f"M={self._benchmark.config.M}, next=round_robin"
                )
                ranked_all = sorted(
                    population.individuals,
                    key=lambda ind: ind.trueskill_mu,
                    reverse=True,
                )
                return self._run_round_robin(population, ranked_all, generation, gen_dir=gen_dir)

        # ── 阶段 1：基准筛选（二分种子） ──
        ranked_all = self._benchmark.evaluate(population, generation=generation, gen_dir=gen_dir)

        return self._run_round_robin(population, ranked_all, generation, gen_dir=gen_dir)

    def _should_skip_touchstone(self, survivor_count: int) -> bool:
        max_discovery_games = (survivor_count - 1) * self._benchmark.config.discovery_n_games
        return max_discovery_games < self._benchmark.config.M

    def _run_round_robin(
        self,
        population: Population,
        ranked_all: list,
        generation: int,
        gen_dir: str = None,
    ) -> tuple:
        # ── 阶段 2：循环赛 ──
        top_n = self._round_robin.config.top_n
        top_individuals = ranked_all[:min(top_n, len(ranked_all))]

        seeds, elites = self._round_robin.run(
            top_individuals=top_individuals,
            num_seeds=self.num_seeds,
            elitism_count=self.elitism_count,
            dedup_config=self._dedup_config,
            generation=generation,
            gen_dir=gen_dir,
        )

        # 更新种群统计
        ranked_all = sorted(ranked_all, key=lambda ind: ind.trueskill_mu, reverse=True)
        population.best_rating = ranked_all[0].trueskill_mu if ranked_all else 0.0
        mu_values = [ind.trueskill_mu for ind in ranked_all]
        population.mean_rating = sum(mu_values) / max(len(mu_values), 1)

        return ranked_all, seeds, elites

    @property
    def benchmark(self) -> TouchstoneSelector:
        return self._benchmark

    @property
    def round_robin(self):
        return self._round_robin
