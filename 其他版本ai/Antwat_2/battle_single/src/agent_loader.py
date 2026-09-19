import os
import sys
import random
from typing import Optional, Any, List

from .battle_single_logger import BattleLogger, LogLevel
from .medium_rule_ai import MediumRuleAI, BuiltinAgent


class BasicRandomAI(BuiltinAgent):
    def __init__(self, seed: int = None):
        self.base_seed = seed
        self.rng = random.Random(seed)
        self.episode_count = 0
        self.positions = self._load_positions()
        self.built_positions = set()

    @staticmethod
    def _load_positions():
        try:
            from SDK.utils.constants import HIGHLAND_CELLS
            return {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}
        except Exception:
            return {0: [], 1: []}

    def choose_operations(self, state: Any, player: int) -> List[Any]:
        from SDK.backend.model import Operation, OperationType

        try:
            player_coins = 0
            if hasattr(state, 'players') and len(state.players) > player:
                player_coins = state.players[player].coins
            elif hasattr(state, 'coins'):
                coins = getattr(state, 'coins', [0, 0])
                player_coins = coins[player] if isinstance(coins, (list, tuple)) else coins

            all_positions = self.positions.get(player, [])
            available = [p for p in all_positions if p not in self.built_positions]
            if player_coins >= 15 and len(available) > 0 and self.rng.random() < 0.5:
                pos = available[self.rng.randint(0, len(available) - 1)]
                self.built_positions.add(pos)
                return [Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])]

            return []

        except Exception:
            return []

    def reset_for_episode(self, episode_num: int):
        if self.base_seed is not None:
            self.rng = random.Random(self.base_seed + episode_num)
        self.episode_count = episode_num
        self.built_positions.clear()


class BasicTowerAI(BuiltinAgent):
    def __init__(self):
        self.positions = self._load_positions()
        self.next_tower_index = 0
        self.built_positions = set()

    @staticmethod
    def _load_positions():
        try:
            from SDK.utils.constants import HIGHLAND_CELLS
            return {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}
        except Exception:
            return {0: [], 1: []}

    def choose_operations(self, state: Any, player: int) -> List[Any]:
        from SDK.backend.model import Operation, OperationType

        try:
            player_coins = 0
            if hasattr(state, 'players') and len(state.players) > player:
                player_coins = state.players[player].coins
            elif hasattr(state, 'coins'):
                coins = getattr(state, 'coins', [0, 0])
                player_coins = coins[player] if isinstance(coins, (list, tuple)) else coins

            all_positions = self.positions.get(player, [])
            available = [p for p in all_positions if p not in self.built_positions]
            if player_coins >= 30 and len(available) > 0:
                pos = available[self.next_tower_index % len(available)]
                self.next_tower_index += 1
                self.built_positions.add(pos)
                return [Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])]

            return []

        except Exception:
            return []

    def reset_for_episode(self, episode_num: int):
        all_positions = self.positions.get(0, [])
        if all_positions:
            self.next_tower_index = episode_num % len(all_positions)
        self.built_positions.clear()


BUILTIN_AGENTS = {
    'BasicRandomAI': BasicRandomAI,
    'BasicTowerAI': BasicTowerAI,
    'MediumRuleAI': MediumRuleAI,
}


def is_builtin_agent(agent_name: str) -> bool:
    return agent_name in BUILTIN_AGENTS


def create_builtin_agent(agent_name: str) -> Optional[BuiltinAgent]:
    if agent_name not in BUILTIN_AGENTS:
        return None

    try:
        agent_class = BUILTIN_AGENTS[agent_name]
        agent = agent_class()
        return agent
    except Exception:
        return None


BASELINES_PATH = os.environ.get('BASELINES_PATH', '/root/autodl-tmp/AntWar/baselines')


class AgentLoader:
    def __init__(self, logger: BattleLogger = None):
        self.logger = logger
    
    def _info(self, category, message):
        if self.logger:
            self.logger.info(category, message)
    
    def _warning(self, category, message):
        if self.logger:
            self.logger.warning(category, message)
    
    def _debug(self, category, message):
        if self.logger:
            self.logger.debug(category, message)
    
    def _log_agent_load(self, agent_name, success, error_msg=None):
        if self.logger:
            self.logger.log_agent_load(agent_name, success, error_msg)

    def find_baselines_path(self) -> str:
        candidate_paths = [
            BASELINES_PATH,
            os.path.join(os.path.dirname(__file__), '../../baselines'),
            '/root/autodl-tmp/AntWar/baselines',
            '/autodl-tmp/AntWar/baselines',
        ]

        for path in candidate_paths:
            if os.path.exists(path) and os.path.isdir(path):
                return path

        raise RuntimeError(f"无法找到 baselines 目录，已尝试: {candidate_paths}")

    def get_available_agents(self, baseline_path: str = None) -> List[str]:
        if baseline_path is None:
            baseline_path = self.find_baselines_path()

        available = []
        if os.path.exists(baseline_path):
            for item in os.listdir(baseline_path):
                item_path = os.path.join(baseline_path, item)
                if os.path.isdir(item_path):
                    ai_file = os.path.join(item_path, 'ai.py')
                    if os.path.exists(ai_file):
                        available.append(item)
        else:
            self._warning('load', f"baselines 目录不存在: {baseline_path}")

        return sorted(available)

    def load_agent(self, agent_name: str, baseline_path: str = None) -> Optional[Any]:
        if is_builtin_agent(agent_name):
            agent = create_builtin_agent(agent_name)
            if agent:
                self._log_agent_load(agent_name, True)
                return agent
            self._log_agent_load(agent_name, False, f"内置 agent 创建失败")
            return None

        if baseline_path is None:
            baseline_path = self.find_baselines_path()

        agent_path = os.path.join(baseline_path, agent_name)

        if not os.path.exists(agent_path):
            self._log_agent_load(agent_name, False, f"目录不存在: {agent_path}")
            return None

        ai_file = os.path.join(agent_path, 'ai.py')
        if not os.path.exists(ai_file):
            self._log_agent_load(agent_name, False, f"缺少 ai.py 文件")
            return None

        original_sys_path = sys.path.copy()
        original_modules = {}

        modules_to_remove = []
        for key in list(sys.modules.keys()):
            if key in ['ai', 'common', 'SDK', 'AI'] or \
               key.startswith('ai.') or key.startswith('common.') or \
               key.startswith('SDK.') or key.startswith('AI.'):
                modules_to_remove.append(key)

        project_sdk_paths = [
            '/root/autodl-tmp/AntWar/Ant-Game/SDK',
            os.path.join(os.path.dirname(__file__), '../../Ant-Game/SDK'),
            os.path.join(os.path.dirname(__file__), '../../../Ant-Game/SDK'),
        ]
        
        project_sdk_path = None
        for path in project_sdk_paths:
            if os.path.exists(path):
                project_sdk_path = path
                break
        
        if project_sdk_path:
            sdk_parent_path = os.path.dirname(project_sdk_path)
            sys.path.insert(0, sdk_parent_path)
            sys.path.insert(0, project_sdk_path)
            self._debug('sdk', f"使用项目 SDK: {project_sdk_path}")
        
        sys.path.insert(0, agent_path)

        for mod in modules_to_remove:
            if mod in sys.modules:
                original_modules[mod] = sys.modules[mod]
                del sys.modules[mod]

        try:
            os.chdir(agent_path)

            try:
                from ai import create_agent
                agent = create_agent()
            except ImportError:
                from ai import AI
                agent = AI()

            import torch
            for attr_name in ['network', 'neural_agent', 'model', 'nn_model']:
                if hasattr(agent, attr_name):
                    model_obj = getattr(agent, attr_name)
                    if hasattr(model_obj, 'to'):
                        model_obj.to('cpu')
                        if hasattr(agent, 'device'):
                            agent.device = torch.device('cpu')
                        break

            self._log_agent_load(agent_name, True)
            return agent

        except Exception as e:
            import traceback
            error_msg = str(e) + "\n" + traceback.format_exc()
            self._log_agent_load(agent_name, False, error_msg)
            return None

        finally:
            sys.path = original_sys_path
            for mod, module in original_modules.items():
                sys.modules[mod] = module