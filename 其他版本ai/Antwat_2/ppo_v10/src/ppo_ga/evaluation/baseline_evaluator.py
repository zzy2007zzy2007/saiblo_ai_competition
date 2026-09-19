from typing import List, Dict, Any, Optional
from loguru import logger

from ..evolution.population_manager import Individual
from ..battle.battle_simulator import BattleSimulator
from ..battle.battle_config import GABattleConfig
from ..battle.ga_war_agent import GAWarAgent
from ..battle.agent_loader import AgentLoader
from ..battle.result_aggregator import aggregate_results, aggregate_illegal_stats, save_round_logs
from ..genome.genome import load_genome_to_network
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork


# 默认基线 Agent 列表
DEFAULT_BASELINE_AGENTS = ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"]


class BaselineEvaluator:
    """基线验证器 —— 种子与固定 baseline AI 对战，验证绝对质量"""

    def __init__(
        self,
        config: GABattleConfig,
        device: str = "cuda",
        hidden_dim: int = 256,
        enabled: bool = True,
        interval: int = 1,
        n_battles: int = 12,
        baseline_agents: Optional[List[str]] = None,
    ):
        self.config = config
        self.device = device
        self.hidden_dim = hidden_dim
        self.enabled = enabled
        self.interval = interval
        self.n_battles = n_battles
        self.baseline_agents = baseline_agents or DEFAULT_BASELINE_AGENTS
        self._simulator = BattleSimulator(config)

    def should_evaluate(self, generation: int) -> bool:
        """判断当前代是否需要执行基线验证"""
        if not self.enabled:
            return False
        return generation % self.interval == 0

    def evaluate_seeds(
        self,
        seeds: List[Individual],
        generation: int,
        gen_dir: str = None,
    ) -> Dict[str, Dict[str, Any]]:
        """对种子选手执行基线验证。"""
        if not self.should_evaluate(generation):
            if self.enabled:
                logger.info(
                    f"[phase=baseline_battle][gen={generation}] skip: "
                    f"reason=interval, interval={self.interval}"
                )
            return {}

        total_games = len(seeds) * len(self.baseline_agents) * self.n_battles * 2
        logger.info(
            f"[phase=baseline_battle][gen={generation}] start: "
            f"seeds={len(seeds)}, agents={len(self.baseline_agents)}, "
            f"n_battles={self.n_battles}, games={total_games}"
        )

        # 收集所有种子 vs 所有 baseline agent 的原始对战结果（含回合详情）
        all_raw_results = []

        results = {}
        for seed in seeds:
            seed_results, seed_raw_results = self._evaluate_single_seed(seed, generation)
            results[seed.id] = seed_results
            all_raw_results.extend(seed_raw_results)

        # 保存完整回合日志
        if gen_dir is not None and all_raw_results:
            save_round_logs("baseline", generation, gen_dir, all_raw_results)

        logger.info(
            f"[phase=baseline_battle][gen={generation}] done: "
            f"seeds={len(results)}, agents={len(self.baseline_agents)}, games={total_games}"
        )
        return results

    def _evaluate_single_seed(
        self,
        seed: Individual,
        generation: int,
    ) -> tuple:
        """评估单个种子 vs 所有 baseline Agent。

        Returns:
            (results, raw_results):
            - results: 按 baseline agent 聚合的统计结果 dict。
            - raw_results: 包含 home/away 字段的原始对战结果列表（用于保存回合日志）。
        """
        import torch

        device = torch.device(self.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.hidden_dim,
        ).to(device)
        load_genome_to_network(network, seed.genome)
        network.eval()

        ga_agent = GAWarAgent(player_id=0, network=network, device=device)

        raw_results = []
        results = {}
        for baseline_name in self.baseline_agents:
            baseline_agent = AgentLoader.load(baseline_name, player_id=1)

            battle_results = self._simulator.run_battles(
                ga_agent, baseline_agent, self.n_battles
            )

            # 注入 home/away 字段后收集原始结果
            for br in battle_results:
                br["home"] = seed.id
                br["away"] = baseline_name
                raw_results.append(br)

            # debug 日志：前 5 回合采样
            for br in battle_results:
                if "error" in br:
                    logger.debug(
                        f"[phase=baseline_battle][gen={generation}] battle_error: "
                        f"seed={seed.id}, agent={baseline_name}, "
                        f"error={br.get('error', 'unknown')}"
                    )
                    continue
                logger.debug(
                    f"[phase=baseline_battle][gen={generation}] battle: "
                    f"seed={seed.id}, agent={baseline_name}, "
                    f"result={br.get('result')}, first_player={br.get('first_player')}, "
                    f"rounds={br.get('total_rounds')}, "
                    f"hp={br.get('final_hp', {})}, duration={br.get('duration', 0):.1f}s"
                )
                rounds_data = br.get("rounds", [])
                for rd in rounds_data[-5:]:
                    logger.debug(
                        f"[phase=baseline_battle][gen={generation}] round_sample: "
                        f"seed={seed.id}, agent={baseline_name}, round={rd.get('round')}, "
                        f"a1_ops={rd.get('agent1_ops')}, a2_ops={rd.get('agent2_ops')}, "
                        f"a1_hp={rd.get('agent1_hp')}, a2_hp={rd.get('agent2_hp')}, "
                        f"a1_towers={rd.get('agent1_towers')}, a2_towers={rd.get('agent2_towers')}"
                    )

            aggregated = aggregate_results(battle_results)
            # 非法动作统计
            illegal_stats = aggregate_illegal_stats(battle_results)
            aggregated["total_illegal"] = illegal_stats["total_illegal"]
            results[baseline_name] = aggregated

            win_rate = aggregated.get("agent1_wins", 0) / max(aggregated.get("total_battles", 1), 1)
            avg_rounds = aggregated.get("avg_rounds", 0)
            logger.info(
                f"[phase=baseline_battle][gen={generation}] matchup_done: "
                f"seed={seed.id}, agent={baseline_name}, win_rate={win_rate:.2%}, "
                f"wins={aggregated.get('agent1_wins', 0)}, "
                f"losses={aggregated.get('agent2_wins', 0)}, "
                f"draws={aggregated.get('draws', 0)}, "
                f"avg_rounds={avg_rounds:.1f}, "
                f"illegal={aggregated.get('total_illegal', 0)}"
            )

        # 跨 agent 汇总
        win_rates = []
        avg_rounds_list = []
        for agg in results.values():
            wr = agg.get("agent1_wins", 0) / max(agg.get("total_battles", 1), 1)
            win_rates.append(wr)
            avg_rounds_list.append(agg.get("avg_rounds", 0))
        avg_win_rate = sum(win_rates) / max(len(win_rates), 1)
        mean_avg_rounds = sum(avg_rounds_list) / max(len(avg_rounds_list), 1)
        logger.info(
            f"[phase=baseline_battle][gen={generation}] seed_done: "
            f"seed={seed.id}, avg_win_rate={avg_win_rate:.2%}, "
            f"avg_rounds={mean_avg_rounds:.1f}"
        )

        # 该 seed 跨 agent 的非法动作汇总
        illegal_total = sum(agg.get("total_illegal", 0) for agg in results.values())
        if illegal_total > 0:
            logger.info(
                f"[phase=baseline_battle][gen={generation}] seed_illegal: "
                f"seed={seed.id}, total={illegal_total}"
            )

        return results, raw_results
