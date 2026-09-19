from typing import Any, Dict, List, Optional
import time as time_module

from loguru import logger

from .agent_loader import AgentLoader
from .battle_config import BaselineBattleConfig
from .battle_simulator import BattleSimulator
from .result_aggregator import aggregate_results
from .report_generator import BattleReporter
from ..config.path_config import PathConfig


class BattleCoordinator:
    """对战协调器 - 一站式对战评估入口。

    整合配置管理、智能体加载、对战执行、结果聚合和报告生成。
    """

    def __init__(
        self,
        config: BaselineBattleConfig,
        path_config: PathConfig,
        eval_logger: Any = None,
    ):
        self.config = config
        self.path_config = path_config
        self._eval_logger = eval_logger

        AgentLoader.set_baselines_path(str(path_config.baselines_path))

        self._simulator = BattleSimulator(config)
        self._reporter = BattleReporter()

    def run_ppo_evaluation_with_agent(
        self,
        ppo_agent,
        baseline_agents: List[str],
        n_battles: Optional[int] = None,
        episode: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """执行 PPO 智能体与基线智能体的对战评估。

        Args:
            episode: 当前 episode 编号，用于回合日志上下文。为 None 时不写回合日志。

        Returns:
            {baseline_name: {wins, losses, draws, win_rate, ...}, ...}
        """
        results = {}
        n_battles = n_battles or getattr(self.config, "n_battles", 8)

        AgentLoader.validate_baselines(baseline_agents)

        for baseline_name in baseline_agents:
            try:
                baseline_agent = AgentLoader.load(baseline_name, player_id=1)

                self._init_logger("PPO", baseline_name)

                battle_results = self._simulator.run_battles(
                    ppo_agent, baseline_agent, n_battles
                )

                aggregated = aggregate_results(battle_results)
                results[baseline_name] = aggregated

                error_count = aggregated.get("error_battles", 0)
                if error_count > 0:
                    logger.warning(
                        f"vs {baseline_name}: {error_count}/{aggregated['total_battles']} "
                        f"battles had errors"
                    )

                report = self._reporter.generate_report(
                    aggregated, "PPO", baseline_name
                )
                logger.info(f"\n{report}")

                # 写回合级日志
                if self._eval_logger is not None and episode is not None:
                    self._write_round_logs(
                        battle_results, baseline_name, episode
                    )

            except Exception as e:
                logger.error(f"Failed to evaluate vs {baseline_name}: {e}")
                results[baseline_name] = {
                    "error": str(e),
                    "agent1_wins": 0,
                    "agent2_wins": 0,
                    "draws": 0,
                }

        return results

    def _write_round_logs(
        self,
        battle_results: List[Dict[str, Any]],
        baseline_name: str,
        episode: int,
    ) -> None:
        """将每局回合数据写入回合日志。

        agent1 始终是 PPO agent，agent2 始终是 baseline agent。
        agent1_name/agent2_name 字段记录了 agent 的名字用于识别身份。
        """
        for battle_idx, result in enumerate(battle_results):
            rounds = result.get("rounds")
            if not rounds:
                continue
            for rd in rounds:
                agent1_ops_str = _compact_ops(rd.get("agent1_ops", []))
                agent2_ops_str = _compact_ops(rd.get("agent2_ops", []))
                self._eval_logger.write_round(
                    episode=episode,
                    round_num=rd.get("round", 0),
                    baseline=baseline_name,
                    battle_idx=battle_idx,
                    time=time_module.time(),
                    agent1_name=rd.get("agent1_name", ""),
                    agent2_name=rd.get("agent2_name", ""),
                    agent1_ops=agent1_ops_str,
                    agent2_ops=agent2_ops_str,
                    agent1_hp=rd.get("agent1_hp"),
                    agent2_hp=rd.get("agent2_hp"),
                    agent1_hp_before=rd.get("agent1_hp_before"),
                    agent2_hp_before=rd.get("agent2_hp_before"),
                    agent1_hp_dmg=rd.get("agent1_hp_dmg"),
                    agent2_hp_dmg=rd.get("agent2_hp_dmg"),
                    agent1_twr_hp_before=rd.get("agent1_twr_hp_before"),
                    agent2_twr_hp_before=rd.get("agent2_twr_hp_before"),
                    agent1_twr_hp_after=rd.get("agent1_twr_hp_after"),
                    agent2_twr_hp_after=rd.get("agent2_twr_hp_after"),
                    agent1_twr_hp_dmg=rd.get("agent1_twr_hp_dmg"),
                    agent2_twr_hp_dmg=rd.get("agent2_twr_hp_dmg"),
                    agent1_coins=rd.get("agent1_coins"),
                    agent2_coins=rd.get("agent2_coins"),
                    agent1_coins_before=rd.get("agent1_coins_before"),
                    agent2_coins_before=rd.get("agent2_coins_before"),
                    agent1_ops_cost=rd.get("agent1_ops_cost"),
                    agent2_ops_cost=rd.get("agent2_ops_cost"),
                    agent1_coins_earned=rd.get("agent1_coins_earned"),
                    agent2_coins_earned=rd.get("agent2_coins_earned"),
                    agent1_towers=rd.get("agent1_towers"),
                    agent2_towers=rd.get("agent2_towers"),
                )


    def run_batch(
        self,
        agent1: Any,
        agent2: Any,
        n_battles: int,
    ) -> Dict[str, Any]:
        """执行两个指定智能体之间的对战"""
        self._init_logger("Agent1", "Agent2")

        battle_results = self._simulator.run_battles(agent1, agent2, n_battles)

        aggregated = aggregate_results(battle_results)
        return aggregated

    def _init_logger(self, agent1_name: str, agent2_name: str) -> None:
        """初始化对战日志记录器"""
        logger.info(f"Starting battle evaluation: {agent1_name} vs {agent2_name}")


def _compact_ops(ops_list: List[Dict]) -> str:
    """将操作列表压缩为紧凑字符串，便于日志阅读。
    例如: [{"type": "BUILD_TOWER", "x": 5, "y": 3}] → "BUILD_TOWER(5,3)"
    """
    parts = []
    for op in ops_list:
        t = op.get("type", "?")
        x = op.get("x")
        y = op.get("y")
        if x is not None and y is not None:
            parts.append(f"{t}({x},{y})")
        elif x is not None:
            parts.append(f"{t}({x})")
        else:
            parts.append(t)
    return "+".join(parts)
