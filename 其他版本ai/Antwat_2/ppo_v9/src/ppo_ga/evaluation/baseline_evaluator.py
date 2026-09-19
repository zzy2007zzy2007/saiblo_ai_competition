from typing import List, Dict, Any, Optional
from loguru import logger

from ..evolution.population_manager import Individual
from ..battle.battle_simulator import BattleSimulator
from ..battle.battle_config import GABattleConfig
from ..battle.ga_war_agent import GAWarAgent
from ..battle.agent_loader import AgentLoader
from ..battle.result_aggregator import aggregate_results
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
        enable_auxiliary: bool = True,
        enabled: bool = True,
        interval: int = 1,
        n_battles: int = 12,
        baseline_agents: Optional[List[str]] = None,
    ):
        self.config = config
        self.device = device
        self.hidden_dim = hidden_dim
        self.enable_auxiliary = enable_auxiliary
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
    ) -> Dict[str, Dict[str, Any]]:
        """对种子选手执行基线验证。"""
        if not self.should_evaluate(generation):
            if self.enabled:
                logger.info(f"Baseline evaluation skipped (gen {generation}, interval={self.interval})")
            return {}

        logger.info(
            f"Baseline evaluation: {len(seeds)} seeds x {len(self.baseline_agents)} agents, n_battles={self.n_battles}"
        )

        results = {}
        for seed in seeds:
            seed_results = self._evaluate_single_seed(seed)
            results[seed.id] = seed_results
        return results

    def _evaluate_single_seed(self, seed: Individual) -> Dict[str, Dict[str, Any]]:
        """评估单个种子 vs 所有 baseline Agent"""
        import torch

        device = torch.device(self.device)
        network = AntWarPolicyValueNetwork(
            hidden_dim=self.hidden_dim,
            enable_auxiliary=self.enable_auxiliary,
        ).to(device)
        load_genome_to_network(network, seed.genome)
        network.eval()

        ga_agent = GAWarAgent(player_id=0, network=network, device=device)

        results = {}
        for baseline_name in self.baseline_agents:
            baseline_agent = AgentLoader.load(baseline_name, player_id=1)

            battle_results = self._simulator.run_battles(
                ga_agent, baseline_agent, self.n_battles
            )

            # 回合级别 debug 日志
            for br in battle_results:
                if "error" in br:
                    logger.debug(f"Baseline battle ERROR: {seed.id} vs {baseline_name}: {br.get('error', 'unknown')}")
                    continue
                logger.debug(
                    f"Baseline battle: {seed.id} vs {baseline_name} "
                    f"result={br.get('result')} first_player={br.get('first_player')} "
                    f"rounds={br.get('total_rounds')} "
                    f"hp={br.get('final_hp', {})} duration={br.get('duration', 0):.1f}s"
                )
                rounds_data = br.get("rounds", [])
                for rd in rounds_data[-5:]:
                    logger.debug(
                        f"  Round {rd.get('round')}: "
                        f"a1_ops={rd.get('agent1_ops')} a2_ops={rd.get('agent2_ops')} "
                        f"a1_hp={rd.get('agent1_hp')} a2_hp={rd.get('agent2_hp')} "
                        f"a1_towers={rd.get('agent1_towers')} a2_towers={rd.get('agent2_towers')}"
                    )

            aggregated = aggregate_results(battle_results)
            results[baseline_name] = aggregated

            win_rate = aggregated.get("agent1_wins", 0) / max(aggregated.get("total_battles", 1), 1)
            logger.info(
                f"Seed {seed.id} vs {baseline_name}: "
                f"win_rate={win_rate:.2%} "
                f"({aggregated.get('agent1_wins', 0)}W/"
                f"{aggregated.get('agent2_wins', 0)}L/"
                f"{aggregated.get('draws', 0)}D)"
            )

        # 跨 agent 汇总
        win_rates = []
        for agent_name, agg in results.items():
            wr = agg.get("agent1_wins", 0) / max(agg.get("total_battles", 1), 1)
            win_rates.append(wr)
        avg_win_rate = sum(win_rates) / max(len(win_rates), 1)
        logger.info(f"Seed {seed.id} baseline done: avg_win_rate={avg_win_rate:.2%}")

        return results
