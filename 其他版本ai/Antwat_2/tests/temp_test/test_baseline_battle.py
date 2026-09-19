#!/usr/bin/env python3
"""
测试 baseline 模型与 BasicTowerAI 的对战
- BasicTowerAI 使用项目主 SDK
- 其他 baseline 模型使用自己的 SDK
"""

import os
import sys
import time
import random
from datetime import datetime
import numpy as np


class BattleLogger:
    LOG_LEVEL_DEBUG = 0
    LOG_LEVEL_INFO = 1
    LOG_LEVEL_WARNING = 2
    LOG_LEVEL_ERROR = 3

    def __init__(self, log_level=LOG_LEVEL_INFO):
        self.log_level = log_level

    def _write(self, level_str, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"[{timestamp}] [{level_str}] {message}")

    def debug(self, message):
        if self.log_level <= self.LOG_LEVEL_DEBUG:
            self._write("DEBUG", message)

    def info(self, message):
        if self.log_level <= self.LOG_LEVEL_INFO:
            self._write("INFO", message)

    def warning(self, message):
        if self.log_level <= self.LOG_LEVEL_WARNING:
            self._write("WARNING", message)

    def error(self, message):
        if self.log_level <= self.LOG_LEVEL_ERROR:
            self._write("ERROR", message)


class AgentWrapper:
    """包装 agent 包装器，确保 agent 使用正确的 SDK"""

    def __init__(self, agent, model_name, model_path, sdk_path=None):
        self.agent = agent
        self.model_name = model_name
        self.model_path = model_path
        self.sdk_path = sdk_path
        self.original_sys_path = None

    def _setup_sdk(self):
        """设置 SDK 路径"""
        self.original_sys_path = sys.path.copy()
        if self.sdk_path:
            new_path = []
            new_path.insert(0, self.sdk_path)
            new_path.insert(0, self.model_path)
            
            for p in self.original_sys_path:
                if "site-packages" in p or p == "" or "python3.12" in p or "lib/python" in p:
                    new_path.append(p)
            sys.path = new_path

    def _restore_sdk(self):
        """恢复 SDK 路径"""
        if self.original_sys_path:
            sys.path = self.original_sys_path

    def choose_operations(self, state, player):
        """获取操作时设置正确的 SDK"""
        self._setup_sdk()
        try:
            result = self.agent.choose_operations(state, player)
            self._restore_sdk()
            return result
        except Exception as e:
            self._restore_sdk()
            raise

    def choose_bundle(self, state, player, bundles):
        self._setup_sdk()
        try:
            result = self.agent.choose_bundle(state, player, bundles)
            self._restore_sdk()
            return result
        except Exception as e:
            self._restore_sdk()
            raise

    def list_bundles(self, state, player):
        self._setup_sdk()
        try:
            result = self.agent.list_bundles(state, player)
            self._restore_sdk()
            return result
        except Exception as e:
            self._restore_sdk()
            raise


class BattleSimulator:
    BATTLE_PRINT_INTERVAL = 20

    def __init__(self, logger, base_dir):
        self.logger = logger
        self.base_dir = base_dir
        self.antwar_path = os.path.join(base_dir, "saiblo-antwar-sdk-python")
        self._load_main_sdk()

    def _load_main_sdk(self):
        """加载主 SDK（用于 BasicTowerAI）"""
        if os.path.exists(self.antwar_path):
            sys.path.insert(0, self.antwar_path)
        self.logger.info(f"✓ 加载主 SDK: {self.antwar_path}")

    def _load_agent_with_own_sdk(self, model_name, model_type=None):
        """加载 agent 并设置其 SDK"""
        model_path = os.path.join(self.base_dir, "baselines", model_name)
        sdk_path = os.path.join(model_path, "SDK")

        if not os.path.exists(model_path):
            self.logger.error(f"✗ 模型路径不存在: {model_path}")
            return None

        original_sys_path = sys.path.copy()
        new_path = []

        if os.path.exists(sdk_path):
            new_path.insert(0, sdk_path)

        new_path.insert(0, model_path)

        for p in original_sys_path:
            if "site-packages" in p or p == "" or "python3.12" in p or "lib/python" in p:
                new_path.append(p)

        sys.path = new_path

        try:
            ai_path = os.path.join(model_path, "ai.py")
            if os.path.exists(ai_path):
                from ai import AI
                agent = AI()
                self.logger.info(f"✓ 加载 AI: {model_name}")
                sys.path = original_sys_path
                return AgentWrapper(agent, model_name, model_path, sdk_path)
            else:
                self.logger.error(f"✗ 找不到 ai.py: {model_name}")
                sys.path = original_sys_path
                return None
        except Exception as e:
            import traceback
            self.logger.error(f"✗ 加载失败: {model_name}: {e}")
            self.logger.error(traceback.format_exc())
            sys.path = original_sys_path
            return None

    def _get_basic_tower_ai(self):
        """获取 BasicTowerAI（使用主 SDK）"""
        from SDK.backend.state import PythonBackendState
        from SDK.backend.core import load_backend
        from SDK.backend.model import Operation as BackendOperation
        from SDK.utils.constants import OperationType as BackendOperationType, MAX_ROUND
        from SDK.utils.actions import ActionBundle
        from antwar.coord import Coord
        from antwar.protocol import build_tower_op

        class BasicTowerAI:
            def choose_operations(self, state, player):
                positions = [Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12)]
                for pos in positions:
                    if state.coins[player] >= 30:
                        op = build_tower_op(pos)
                        return [BackendOperation(
                            op_type=BackendOperationType(op.type.value),
                            arg0=op.arg0, arg1=op.arg1
                        )]
                return []

        return BasicTowerAI()

    def run_battle(self, agent_wrapper, agent_name, episodes=2, max_rounds=100):
        """运行对战"""
        from SDK.backend.state import PythonBackendState
        from SDK.backend.core import load_backend
        from SDK.backend.model import Operation as BackendOperation
        from SDK.utils.constants import OperationType as BackendOperationType, MAX_ROUND

        opponent = self._get_basic_tower_ai()
        opponent_name = "BasicTowerAI"

        agent_wins = 0
        basic_wins = 0
        draws = 0
        total_rounds = 0

        for episode in range(episodes):
            battle_result = self._run_single_battle(
                agent_wrapper, agent_name, opponent, opponent_name,
                episode, max_rounds
            )

            total_rounds += battle_result['rounds_played']

            if battle_result['result'] == 'win':
                agent_wins += 1
                self.logger.info(f"✓ 第 {episode + 1} 场: {agent_name} 胜")
            elif battle_result['result'] == 'loss':
                basic_wins += 1
                self.logger.info(f"✗ 第 {episode + 1} 场: {agent_name} 负")
            else:
                draws += 1
                self.logger.info(f"— 第 {episode + 1} 场: 平局")

        return {
            'agent_name': agent_name,
            'agent_wins': agent_wins,
            'basic_wins': basic_wins,
            'draws': draws,
            'total_rounds': total_rounds,
            'win_rate': agent_wins / episodes if episodes > 0 else 0
        }

    def _run_single_battle(self, agent_wrapper, agent_name, opponent, opponent_name, episode, max_rounds):
        """运行单场对战"""
        from SDK.backend.state import PythonBackendState
        from SDK.backend.core import load_backend
        from SDK.utils.constants import MAX_ROUND

        backend = load_backend(prefer_native=False)
        game_state = backend.initial_state(seed=episode)
        state = PythonBackendState(game_state)

        round_count = 0
        start_time = time.time()

        while not state.terminal and round_count < max_rounds:
            round_count += 1

            try:
                ops_agent = agent_wrapper.choose_operations(state, 0)
            except Exception as e:
                self.logger.error(f"获取 {agent_name} 操作失败: {e}")
                ops_agent = []

            try:
                ops_opponent = opponent.choose_operations(state, 1)
            except Exception as e:
                self.logger.error(f"获取 {opponent_name} 操作失败: {e}")
                ops_opponent = []

            state.resolve_turn(ops_agent, ops_opponent)

            if round_count % self.BATTLE_PRINT_INTERVAL == 0:
                hp_info = f"HP: {agent_name}={state.bases[0].hp}, {opponent_name}={state.bases[1].hp}"
                self.logger.debug(f"回合 {round_count}: {hp_info}")

        result = self._determine_result(state)
        duration = time.time() - start_time

        return {
            'result': result,
            'rounds_played': round_count,
            'duration': duration,
            'final_hp': {'agent': state.bases[0].hp, 'opponent': state.bases[1].hp}
        }

    def _determine_result(self, state):
        """确定对战结果"""
        if state.winner == 0:
            return 'win'
        elif state.winner == 1:
            return 'loss'
        else:
            return 'draw'


def main():
    base_dir = '/root/autodl-tmp/AntWar'

    logger = BattleLogger(log_level=BattleLogger.LOG_LEVEL_INFO)
    logger.info("=" * 70)
    logger.info("Baseline 模型对战测试")
    logger.info("=" * 70)

    simulator = BattleSimulator(logger, base_dir)

    models = ["ann_593", "gen99", "gen199"]
    results = {}

    for model_name in models:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"{'=' * 70}")

        agent_wrapper = simulator._load_agent_with_own_sdk(model_name)
        if agent_wrapper is None:
            logger.error(f"✗ 跳过 {model_name} 加载失败")
            continue

        result = simulator.run_battle(agent_wrapper, model_name, episodes=2, max_rounds=100)
        results[model_name] = result

        logger.info(f"\n{'=' * 70}")
        logger.info(f"{model_name} vs BasicTowerAI")
        logger.info(f"{'=' * 70}")
        logger.info(f"胜场: {result['agent_wins']}")
        logger.info(f"负场: {result['basic_wins']}")
        logger.info(f"平局: {result['draws']}")
        logger.info(f"胜率: {result['win_rate'] * 100:.1f}%")
        logger.info(f"总回合数: {result['total_rounds']}")

    logger.info(f"\n{'=' * 70}")
    logger.info("最终汇总")
    logger.info(f"{'=' * 70}")

    for model_name, result in results.items():
        logger.info(f"{model_name}: {result['agent_wins']}胜/{result['basic_wins']}负/{result['draws']}平, 胜率 {result['win_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
