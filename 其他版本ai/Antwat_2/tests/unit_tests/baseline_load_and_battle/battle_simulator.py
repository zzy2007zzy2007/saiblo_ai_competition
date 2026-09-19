#!/usr/bin/env python3
"""
对战模拟器 - 重构版

基于官方规则的重构版本，解决以下问题：
1. 使用 resolve_turn() 替代 apply_operation() + advance_round()
2. 使用 state.winner 判断胜负
3. 使用 state.terminal 判断游戏结束
4. 使用 MAX_ROUNDS = 512 官方标准
5. 添加详细的日志分级

参考：
- ppo_v3/training/battle_logic.py
- ppo_v3/training/baseline_battle_manager.py
- baselines/ann_593/SDK/utils/constants.py
"""

import os
import sys
import time
from datetime import datetime

REMOTE_ANTWAR_PATH = '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python'
REMOTE_SDK_PATH = '/root/autodl-tmp/AntWar/baselines/ann_593'

antwar_paths = [
    REMOTE_ANTWAR_PATH,
    os.path.join(os.path.dirname(__file__), '../../..', 'saiblo-antwar-sdk-python'),
]
sdk_paths = [
    REMOTE_SDK_PATH,
    os.path.join(os.path.dirname(__file__), '../../..', 'baselines', 'ann_593'),
]

for path in antwar_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        print(f"✓ 加载antwar路径: {path}")
        break

for path in sdk_paths:
    if os.path.exists(path):
        sys.path.insert(0, path)
        print(f"✓ 加载SDK路径: {path}")
        break

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from SDK.backend.model import Operation as BackendOperation
from SDK.utils.constants import OperationType as BackendOperationType, MAX_ROUND
from SDK.utils.actions import ActionBundle

LOG_LEVEL_DEBUG = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_WARNING = 2
LOG_LEVEL_ERROR = 3

BATTLE_PRINT_INTERVAL = 10
DEFAULT_MAX_ROUNDS = MAX_ROUND

class BattleLogger:
    """对战日志记录器"""

    def __init__(self, log_level=LOG_LEVEL_INFO, log_file=None):
        self.log_level = log_level
        self.log_file = log_file
        if log_file:
            self._ensure_dir(os.path.dirname(log_file))

    def _ensure_dir(self, directory):
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _write(self, level_str, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        full_msg = f"[{timestamp}] [{level_str}] {message}"
        print(full_msg)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(full_msg + '\n')
            except Exception:
                pass

    def debug(self, message):
        if self.log_level <= LOG_LEVEL_DEBUG:
            self._write("DEBUG", message)

    def info(self, message):
        if self.log_level <= LOG_LEVEL_INFO:
            self._write("INFO", message)

    def warning(self, message):
        if self.log_level <= LOG_LEVEL_WARNING:
            self._write("WARNING", message)

    def error(self, message):
        if self.log_level <= LOG_LEVEL_ERROR:
            self._write("ERROR", message)


class BasicTowerAI:
    """简单策略：优先建造防御塔（使用SDK.Operation）"""
    def choose_operations(self, state, player):
        from antwar.coord import Coord
        from antwar.protocol import build_tower_op

        positions = [Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12)]
        for pos in positions:
            if state.coins[player] >= 30:
                op = build_tower_op(pos)
                return [BackendOperation(
                    op_type=BackendOperationType(op.type.value),
                    arg0=op.arg0, arg1=op.arg1
                )]
        return []


def load_rule_based_agent(config):
    """加载规则型 AI"""
    base_path = config['base_path']
    module_path = os.path.join(base_path, config['path'])
    sys.path.insert(0, module_path)

    try:
        from ai import AI as RuleBasedAI
        agent = RuleBasedAI()
        print(f"✓ 加载规则型AI: {config['name']}")
        return agent
    except Exception as e:
        print(f"✗ 加载规则型AI失败: {e}")
        return None


def create_operation(op_type, arg0=-1, arg1=-1):
    """创建SDK Operation的辅助函数"""
    return BackendOperation(op_type=BackendOperationType(op_type), arg0=arg0, arg1=arg1)


class BattleSimulator:
    """对战模拟器 - 重构版"""

    def __init__(self, log_level=LOG_LEVEL_INFO, log_dir=None):
        self.log_level = log_level
        self.log_dir = log_dir
        self.logger = BattleLogger(log_level=log_level)

    def battle(self, agent, agent_name, agent_type, opponent=None, opponent_name="BasicTowerAI",
               episodes=2, max_rounds=DEFAULT_MAX_ROUNDS, verbose=True):
        """执行模型与对手的对战

        Args:
            agent: 我方智能体
            agent_name: 我方名称
            agent_type: 'neural_network' 或 'rule_based'
            opponent: 对手智能体（默认BasicTowerAI）
            opponent_name: 对手名称
            episodes: 对战局数
            max_rounds: 最大回合数（默认512）
            verbose: 是否打印详细信息

        Returns:
            dict: 包含 agent_wins, basic_wins, draws, battle_details_list
        """
        if opponent is None:
            opponent = BasicTowerAI()

        self.logger.info(f"=== 开始对战: {agent_name} vs {opponent_name} ===")
        self.logger.info(f"最大回合数: {max_rounds}, 对战局数: {episodes}")

        agent_wins = 0
        basic_wins = 0
        draws = 0
        battle_details_list = []

        for episode in range(episodes):
            battle_details = self._run_single_battle(
                agent, agent_name, agent_type,
                opponent, opponent_name,
                episode, max_rounds, verbose
            )
            battle_details_list.append(battle_details)

            if battle_details['result'] == 'win':
                agent_wins += 1
            elif battle_details['result'] == 'loss':
                basic_wins += 1
            else:
                draws += 1

        summary = {
            'agent_wins': agent_wins,
            'basic_wins': basic_wins,
            'draws': draws,
            'battle_details': battle_details_list
        }

        self.logger.info(f"=== 对战统计: {agent_name} {agent_wins}胜 {basic_wins}负 {draws}平 ===")
        return summary

    def _run_single_battle(self, agent, agent_name, agent_type, opponent, opponent_name,
                          episode, max_rounds, verbose):
        """执行单场对战"""
        start_time = time.time()
        start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if verbose:
            self.logger.info(f"\n--- 第 {episode + 1} 场开始 ---")

        backend = load_backend(prefer_native=False)
        game_state = backend.initial_state(seed=episode)
        state = PythonBackendState(game_state)

        round_count = 0
        battle_details = {
            'episode': episode,
            'agent_name': agent_name,
            'opponent_name': opponent_name,
            'start_time': start_time_str,
            'initial_hp': {'agent': state.bases[0].hp, 'opponent': state.bases[1].hp},
            'rounds': [],
            'final_hp': {'agent': 0, 'opponent': 0},
            'result': 'unknown',
            'end_time': '',
            'duration': 0,
            'total_reward': 0,
            'error': None
        }

        try:
            while not state.terminal and round_count < max_rounds:
                round_count += 1

                ops_agent = self._get_agent_operations(agent, agent_type, state, agent_name)
                ops_opponent = self._get_opponent_operations(opponent, state, opponent_name)

                state.resolve_turn(ops_agent, ops_opponent)

                if round_count % BATTLE_PRINT_INTERVAL == 0:
                    hp_info = f"HP: {agent_name}={state.bases[0].hp}, {opponent_name}={state.bases[1].hp}"
                    coins_info = f"金币: {state.coins[0]}, {state.coins[1]}"
                    self.logger.debug(f"回合 {round_count}: {hp_info}, {coins_info}")

                    battle_details['rounds'].append({
                        'round': round_count,
                        'hp': {'agent': state.bases[0].hp, 'opponent': state.bases[1].hp},
                        'coins': {'agent': state.coins[0], 'opponent': state.coins[1]}
                    })

            battle_details['final_hp'] = {
                'agent': state.bases[0].hp,
                'opponent': state.bases[1].hp
            }
            battle_details['total_rounds'] = round_count

            result = self._determine_result(state)
            battle_details['result'] = result

            if verbose:
                hp_info = f"HP: {agent_name}={state.bases[0].hp}, {opponent_name}={state.bases[1].hp}"
                self.logger.info(f"第 {episode + 1} 场结束: {round_count}回合, {hp_info}, 结果: {result}")

        except Exception as e:
            battle_details['error'] = str(e)
            battle_details['result'] = 'error'
            self.logger.error(f"第 {episode + 1} 场异常: {e}")

        end_time = time.time()
        end_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        battle_details['end_time'] = end_time_str
        battle_details['duration'] = end_time - start_time

        return battle_details

    def _get_agent_operations(self, agent, agent_type, state, agent_name):
        """获取我方智能体的操作"""
        try:
            if agent_type == 'neural_network':
                return self._get_neural_network_operations(agent, state)
            elif hasattr(agent, 'choose_operations'):
                return agent.choose_operations(state, 0)
            elif hasattr(agent, 'list_bundles') and hasattr(agent, 'choose_bundle'):
                return self._get_rule_based_operations(agent, state)
            else:
                self.logger.warning(f"{agent_name} 没有可识别的操作方法")
                return []
        except Exception as e:
            self.logger.error(f"获取{agent_name}操作失败: {e}")
            return []

    def _get_neural_network_operations(self, agent, state):
        """获取神经网络智能体的操作"""
        import torch
        import numpy as np
        from torch.distributions import Categorical
        from SDK.utils.features import FeatureExtractor
        from antwar.coord import Coord
        from antwar.protocol import build_tower_op

        action_mask = np.ones(32, dtype=np.float32)
        feature_extractor = FeatureExtractor()
        observation = feature_extractor.encode_observation(state, 0, action_mask)
        flattened_obs = feature_extractor.flatten_observation(observation)

        if not isinstance(flattened_obs, np.ndarray):
            flattened_obs = np.array(flattened_obs, dtype=np.float32)
        obs_tensor = torch.tensor(flattened_obs, dtype=torch.float32).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            action_probs, _ = agent.network(obs_tensor)

        dist = Categorical(action_probs.cpu())
        action_idx = dist.sample().item()

        if action_idx < 10 and state.coins[0] >= 30:
            positions = [
                Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12),
                Coord(7, 7), Coord(7, 11), Coord(11, 7), Coord(11, 11),
                Coord(9, 4), Coord(9, 14)
            ]
            pos = positions[action_idx]
            op = build_tower_op(pos)
            return [BackendOperation(
                op_type=BackendOperationType(op.type.value),
                arg0=op.arg0, arg1=op.arg1
            )]
        return []

    def _get_rule_based_operations(self, agent, state):
        """获取规则型AI的操作"""
        try:
            bundles = agent.list_bundles(state, 0)
            if not bundles:
                return []

            best_bundle = agent.choose_bundle(state, 0, bundles)
            return list(best_bundle.operations)
        except Exception as e:
            self.logger.error(f"规则型AI获取操作失败: {e}")
            return []

    def _get_opponent_operations(self, opponent, state, opponent_name):
        """获取对手的操作"""
        try:
            if hasattr(opponent, 'choose_operations'):
                return opponent.choose_operations(state, 1)
            elif hasattr(opponent, 'choose_bundle'):
                bundles = opponent.list_bundles(state, 1)
                if bundles:
                    best_bundle = opponent.choose_bundle(state, 1, bundles)
                    return list(best_bundle.operations)
            elif hasattr(opponent, 'list_bundles') and hasattr(opponent, 'choose_bundle'):
                bundles = opponent.list_bundles(state, 1)
                if bundles:
                    best_bundle = opponent.choose_bundle(state, 1, bundles)
                    return list(best_bundle.operations)
        except Exception as e:
            self.logger.error(f"获取{opponent_name}操作失败: {e}")
        return []

    def _determine_result(self, state):
        """根据官方规则判断对战结果"""
        if state.winner == 0:
            return 'win'
        elif state.winner == 1:
            return 'loss'
        elif state.bases[0].hp <= 0 and state.bases[1].hp <= 0:
            return 'mutual_destroy'
        else:
            return 'draw'