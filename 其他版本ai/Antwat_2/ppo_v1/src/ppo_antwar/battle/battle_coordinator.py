import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
from loguru import logger

from .battle_config import BaselineBattleConfig
from .battle_logger import BattleLogger, LogLevel
from .agent_loader import AgentLoader
from .battle_simulator import BattleSimulator
from .result_aggregator import aggregate_results
from .report_generator import generate_report


class BattleCoordinator:
    """
    Baseline 对战协调器 - 核心协调类
    负责协调配置管理、agent加载、对战执行、结果聚合和报告生成
    支持两种初始化方式：
    1. 传入 BaselineBattleConfig（向后兼容）
    2. 同时传入 BaselineBattleConfig 和 PathConfig（推荐）
    """

    def __init__(self, config: BaselineBattleConfig, path_config=None):
        self.config = config
        self.config.validate()
        self._path_config = path_config

        self.logger: Optional[BattleLogger] = None
        self.agent_loader: Optional[AgentLoader] = None
        self.simulator: Optional[BattleSimulator] = None

    def _get_battle_dir(self) -> str:
        """获取对战日志目录路径"""
        if self._path_config is not None:
            return str(self._path_config.battle_dir)
        elif self.config.log_dir is not None:
            return str(self.config.log_dir)
        else:
            raise ValueError("No path configuration available for battle logs")

    def _init_logger(self, agent1_name: str, agent2_name: str):
        """初始化日志记录器"""
        log_level = LogLevel.from_string(self.config.log_level)
        battle_dir = self._get_battle_dir()
        self.logger = BattleLogger(
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            log_level=log_level,
            log_to_console=self.config.log_to_console,
            enable_timing=self.config.enable_timing,
            log_dir=battle_dir
        )
        self.agent_loader = AgentLoader(self.logger, path_config=self._path_config)

    def _init_simulator(self):
        """初始化对战模拟器"""
        if self.logger is None:
            raise ValueError("Logger not initialized")
        self.simulator = BattleSimulator(self.logger, max_rounds=self.config.max_rounds)

    def run_battle(
        self,
        agent1: Any,
        agent2: Any,
        agent1_name: str,
        agent2_name: str,
        num_episodes: int = 1
    ) -> List[Dict]:
        """
        执行完整的对战流程

        Args:
            agent1: Agent1 实例
            agent2: Agent2 实例
            agent1_name: Agent1 名称
            agent2_name: Agent2 名称
            num_episodes: 每种先手顺序的对战局数

        Returns:
            对战结果列表
        """
        if self.logger is None:
            self._init_logger(agent1_name, agent2_name)
        if self.simulator is None:
            self._init_simulator()

        self.logger.info('coordinator', f"Starting battle: {agent1_name} vs {agent2_name}")
        self.logger.info('coordinator', f"Number of episodes: {num_episodes} (each order)")
        self.logger.info('coordinator', f"Parallel workers: {self.config.n_battles}")

        results = self.simulator.run_battles_parallel(
            agent1, agent2,
            agent1_name, agent2_name,
            num_episodes=num_episodes,
            parallel_workers=self.config.n_battles
        )

        return results

    def run_evaluation(
        self,
        agent1_name: str,
        agent2_name: str,
        num_episodes: int = 1,
        agent1: Optional[Any] = None,
        agent2: Optional[Any] = None
    ) -> Dict:
        """
        执行完整的评估流程（从配置加载、agent加载、对战执行、结果聚合、报告生成

        Args:
            agent1_name: Agent1 名称
            agent2_name: Agent2 名称
            num_episodes: 每种先手顺序的对战局数
            agent1: 可选，已加载的 Agent1 实例
            agent2: 可选，已加载的 Agent2 实例

        Returns:
            聚合后的结果
        """
        self._init_logger(agent1_name, agent2_name)
        self._init_simulator()

        # 加载 Agent
        if agent1 is None:
            agent1 = self.agent_loader.load_agent(agent1_name)
        if agent1 is None:
            raise RuntimeError(f"Failed to load agent: {agent1_name}")

        if agent2 is None:
            agent2 = self.agent_loader.load_agent(agent2_name)
        if agent2 is None:
            raise RuntimeError(f"Failed to load agent: {agent2_name}")

        # 执行对战
        results = self.run_battle(
            agent1, agent2, agent1_name, agent2_name, num_episodes
        )

        # 聚合结果
        aggregated = aggregate_results(results)

        # 生成报告
        report = generate_report(aggregated, agent1_name, agent2_name)
        for line in report.split('\n'):
            self.logger.info('report', line)

        # 写入汇总日志
        self.logger.write_summary()

        return aggregated

    def run_ppo_evaluation(
        self,
        ppo_checkpoint_path: str,
        baseline_agents: Optional[List[str]] = None,
        num_episodes: int = 1
    ) -> Dict[str, Dict]:
        """
        执行 PPO Agent 与多个 Baseline Agent 的对战评估

        Args:
            ppo_checkpoint_path: PPO Checkpoint 路径
            baseline_agents: Baseline Agent 列表，默认使用配置中的列表
            num_episodes: 每种先手顺序的对战局数

        Returns:
            每个 Baseline Agent 的对战结果字典
        """
        if baseline_agents is None:
            baseline_agents = self.config.baseline_agents

        if not baseline_agents:
            raise ValueError("No baseline agents specified")

        self._init_logger("PPO_Agent", baseline_agents[0])
        self._init_simulator()

        ppo_agent = self.agent_loader.load_ppo_agent(ppo_checkpoint_path)
        if ppo_agent is None:
            raise RuntimeError(f"Failed to load PPO agent from {ppo_checkpoint_path}")

        return self.run_ppo_evaluation_with_agent(
            ppo_agent, baseline_agents, num_episodes
        )

    def run_ppo_evaluation_with_agent(
        self,
        ppo_agent: Any,
        baseline_agents: Optional[List[str]] = None,
        num_episodes: int = 1
    ) -> Dict[str, Dict]:
        """
        使用直接传递的 PPO Agent 对象执行对战评估
        
        Args:
            ppo_agent: PPO Agent 实例（直接传递，非 checkpoint 路径）
            baseline_agents: Baseline Agent 列表，默认使用配置中的列表
            num_episodes: 每种先手顺序的对战局数
        
        Returns:
            每个 Baseline Agent 的对战结果字典
        """
        if baseline_agents is None:
            baseline_agents = self.config.baseline_agents

        if not baseline_agents:
            raise ValueError("No baseline agents specified")

        if self.logger is None:
            self._init_logger("PPO_Agent", baseline_agents[0])
        if self.simulator is None:
            self._init_simulator()

        if hasattr(ppo_agent, 'eval') and callable(getattr(ppo_agent, 'eval')):
            ppo_agent.eval()
        else:
            for attr_name in ['network', 'model', 'nn_model', 'neural_agent', 'policy']:
                if hasattr(ppo_agent, attr_name):
                    model_obj = getattr(ppo_agent, attr_name)
                    if hasattr(model_obj, 'eval') and callable(getattr(model_obj, 'eval')):
                        model_obj.eval()
                        break

        all_results = {}

        for baseline_agent_name in baseline_agents:
            current_logger = BattleLogger(
                agent1_name="PPO_Agent",
                agent2_name=baseline_agent_name,
                log_level=LogLevel.from_string(self.config.log_level),
                log_to_console=self.config.log_to_console,
                enable_timing=self.config.enable_timing,
                log_dir=self._get_battle_dir()
            )

            self.logger = current_logger
            current_agent_loader = AgentLoader(current_logger, path_config=self._path_config)

            try:
                baseline_agent = current_agent_loader.load_agent(baseline_agent_name)
                if baseline_agent is None:
                    current_logger.error('coordinator', f"Skipping {baseline_agent_name}: load failed")
                    continue

                simulator = BattleSimulator(current_logger, max_rounds=self.config.max_rounds)

                results = simulator.run_battles_parallel(
                    ppo_agent, baseline_agent,
                    "PPO_Agent", baseline_agent_name,
                    num_episodes,
                    parallel_workers=self.config.n_battles
                )

                aggregated = aggregate_results(results)

                report = generate_report(aggregated, "PPO_Agent", baseline_agent_name)
                for line in report.split('\n'):
                    current_logger.info('report', line)

                current_logger.write_summary()
                all_results[baseline_agent_name] = aggregated

            except Exception as e:
                current_logger.error('coordinator', f"Evaluation against {baseline_agent_name} failed: {e}")
                import traceback
                current_logger.error('coordinator', traceback.format_exc())
            finally:
                current_logger.close()

        return all_results

    def on_checkpoint_saved(
        self,
        episode: int,
        checkpoint_path: str
    ) -> None:
        """
        当 Checkpoint 保存时触发对战评估

        设计说明：
        - 作为事件驱动接口，响应 checkpoint 保存事件
        - 自动加载最新 checkpoint 的 PPO Agent
        - 与配置的 baseline agents 执行对战
        - 记录评估结果和统计信息

        Args:
            episode: 当前训练 episode
            checkpoint_path: 保存的 checkpoint 路径
        """
        if self.logger is None:
            self._init_logger("PPO_Agent", "baseline")

        self.logger.info('coordinator', f"收到 checkpoint 保存事件: episode={episode}, path={checkpoint_path}")

        try:
            results = self.run_ppo_evaluation(
                ppo_checkpoint_path=checkpoint_path,
                baseline_agents=self.config.baseline_agents,
                num_episodes=self.config.n_battles // 2
            )

            if results:
                self.logger.info('coordinator', f"对战评估完成 @ episode {episode}")
                for agent_name, result in results.items():
                    total_battles = result.get('agent1_wins', 0) + result.get('agent2_wins', 0) + result.get('draws', 0)
                    if total_battles > 0:
                        win_rate = (result.get('agent1_wins', 0) / total_battles) * 100
                    else:
                        win_rate = 0.0
                    self.logger.info('coordinator', f"  vs {agent_name}: 胜率 {win_rate:.1f}%")

        except Exception as e:
            self.logger.error('coordinator', f"on_checkpoint_saved 执行失败: {e}")
            import traceback
            self.logger.error('coordinator', traceback.format_exc())

    def close(self):
        """清理资源"""
        if self.logger:
            self.logger.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
