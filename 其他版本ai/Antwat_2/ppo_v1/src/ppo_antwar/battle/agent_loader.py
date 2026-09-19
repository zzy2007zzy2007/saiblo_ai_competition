import os
import sys
import random
import traceback
from typing import Optional, Any, List, Dict
from abc import ABC, abstractmethod

from loguru import logger as loguru_logger
from .battle_logger import BattleLogger
from .utils.process_isolation import process_isolation
from .utils.exception_logging import log_exception, get_error_context
from ppo_antwar.config.path_config import PathConfig

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BuiltinAgent(ABC):
    @abstractmethod
    def choose_operations(self, state: Any, player: int) -> List[Any]:
        raise NotImplementedError

    def choose_bundle(self, state: Any, player: int, bundles: Optional[List] = None):
        operations = self.choose_operations(state, player)
        return operations

    def reset_for_episode(self, episode_num: int):
        pass


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
        except ImportError:
            pass
        try:
            from ppo_antwar.sdk.utils.constants import HIGHLAND_CELLS
            return {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}
        except ImportError:
            return {0: [], 1: []}

    def choose_operations(self, state: Any, player: int) -> List[Any]:
        try:
            from SDK.backend.model import Operation, OperationType
        except ImportError:
            try:
                from ppo_antwar.sdk.backend.model import Operation, OperationType
            except ImportError:
                return []

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

        except Exception as e:
            error_context = get_error_context(
                player=player,
                state_type=type(state).__name__,
                has_players_attr=hasattr(state, 'players'),
                has_coins_attr=hasattr(state, 'coins')
            )
            log_exception(None, "BasicRandomAI.choose_operations", error_context, e)
            raise

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
        except ImportError:
            pass
        try:
            from ppo_antwar.sdk.utils.constants import HIGHLAND_CELLS
            return {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}
        except ImportError:
            return {0: [], 1: []}

    def choose_operations(self, state: Any, player: int) -> List[Any]:
        try:
            from SDK.backend.model import Operation, OperationType
        except ImportError:
            try:
                from ppo_antwar.sdk.backend.model import Operation, OperationType
            except ImportError:
                return []

        try:
            player_coins = 0
            if hasattr(state, 'players') and len(state.players) > player:
                player_coins = state.players[player].coins
            elif hasattr(state, 'coins'):
                coins = getattr(state, 'coins', [0, 0])
                player_coins = coins[player] if isinstance(coins, (list, tuple)) else coins

            all_positions = self.positions.get(player, [])
            available = [p for p in all_positions if p not in self.built_positions]
            tower_count = state.tower_count(player) if hasattr(state, 'tower_count') else 0
            needed_cost = state.build_tower_cost(tower_count) if hasattr(state, 'build_tower_cost') else 15
            if player_coins >= needed_cost and len(available) > 0:
                pos = available[self.next_tower_index % len(available)]
                self.next_tower_index += 1
                self.built_positions.add(pos)
                return [Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])]

            return []

        except Exception as e:
            error_context = get_error_context(
                player=player,
                state_type=type(state).__name__,
                has_players_attr=hasattr(state, 'players'),
                has_coins_attr=hasattr(state, 'coins')
            )
            log_exception(None, "BasicTowerAI.choose_operations", error_context, e)
            raise

    def reset_for_episode(self, episode_num: int):
        all_positions = self.positions.get(0, [])
        if all_positions:
            self.next_tower_index = episode_num % len(all_positions)
        self.built_positions.clear()


class MediumRuleAI(BuiltinAgent):

    _WEAPON_COST = {
        'lightning_storm': 90,
        'emp_blaster': 135,
        'deflector': 60,
    }

    _TOWER_L2_COST = 60
    _TOWER_L3_COST = 200
    _BASE_UPGRADE_COST = (200, 250)

    def __init__(self):
        try:
            from SDK.utils.constants import STRATEGIC_BUILD_ORDER, TowerType, SuperWeaponType
        except ImportError:
            try:
                from ppo_antwar.sdk.utils.constants import STRATEGIC_BUILD_ORDER, TowerType, SuperWeaponType
            except ImportError:
                STRATEGIC_BUILD_ORDER = {0: [], 1: []}
                TowerType = None
                SuperWeaponType = None

        self._build_positions = STRATEGIC_BUILD_ORDER
        self._TowerType = TowerType
        self._SuperWeaponType = SuperWeaponType
        self._upgraded_towers = set()
        self._ls_positions = {
            0: [(9, 9), (8, 9), (10, 9)],
            1: [(9, 9), (10, 9), (8, 9)],
        }
        self._emp_positions = {
            0: [(10, 9), (11, 9)],
            1: [(8, 9), (7, 9)],
        }
        self._deflector_positions = {
            0: [(6, 9)],
            1: [(12, 9)],
        }

    def reset_for_episode(self, episode_num: int):
        self._upgraded_towers.clear()

    def choose_operations(self, state: Any, player: int) -> List[Any]:
        try:
            from SDK.backend.model import Operation, OperationType
        except ImportError:
            try:
                from ppo_antwar.sdk.backend.model import Operation, OperationType
            except ImportError:
                return []

        try:
            coins = self._get_coins(state, player)
            tower_count = self._get_tower_count(state, player)

            op = (
                self._try_early_build(state, player, coins, tower_count, Operation, OperationType)
                or self._try_upgrade_gen_speed_l1(state, player, coins, Operation, OperationType)
                or self._try_upgrade_tower_l2(state, player, coins, Operation, OperationType)
                or self._try_upgrade_gen_speed_l2(state, player, coins, Operation, OperationType)
                or self._try_lightning_storm(state, player, coins, Operation, OperationType)
                or self._try_emp(state, player, coins, Operation, OperationType)
                or self._try_mid_build(state, player, coins, tower_count, Operation, OperationType)
                or self._try_upgrade_ant_hp_l1(state, player, coins, Operation, OperationType)
                or self._try_upgrade_tower_l3(state, player, coins, Operation, OperationType)
                or self._try_deflector(state, player, coins, Operation, OperationType)
                or self._try_late_build(state, player, coins, tower_count, Operation, OperationType)
            )
            return [op] if op else []

        except Exception as e:
            error_context = get_error_context(
                player=player,
                state_type=type(state).__name__,
                has_players_attr=hasattr(state, 'players'),
                has_coins_attr=hasattr(state, 'coins'),
            )
            log_exception(None, "MediumRuleAI.choose_operations", error_context, e)
            raise

    def _get_coins(self, state, player):
        if hasattr(state, 'players') and len(state.players) > player:
            return state.players[player].coins
        if hasattr(state, 'coins'):
            coins = getattr(state, 'coins', [0, 0])
            return coins[player] if isinstance(coins, (list, tuple)) else coins
        return 0

    def _get_tower_count(self, state, player):
        if hasattr(state, 'tower_count'):
            return state.tower_count(player)
        if hasattr(state, 'towers'):
            return sum(1 for t in state.towers if hasattr(t, 'player') and t.player == player)
        return 0

    def _get_enemy_tower_count(self, state, player):
        enemy = 1 - player
        return self._get_tower_count(state, enemy)

    def _get_round(self, state):
        if hasattr(state, 'round_index'):
            return state.round_index
        return 0

    def _get_gen_level(self, state, player):
        bases = getattr(state, 'bases', [])
        if bases and len(bases) > player:
            return getattr(bases[player], 'generation_level', 0)
        return 0

    def _get_ant_level(self, state, player):
        bases = getattr(state, 'bases', [])
        if bases and len(bases) > player:
            return getattr(bases[player], 'ant_level', 0)
        return 0

    def _get_build_cost(self, state, tower_count):
        if hasattr(state, 'build_tower_cost'):
            return state.build_tower_cost(tower_count)
        base = 15
        cost = base * (3 ** (tower_count // 2))
        if tower_count % 2 == 1:
            cost *= 2
        return cost

    def _weapon_ready(self, state, player, weapon_key):
        idx_map = {'lightning_storm': 1, 'emp_blaster': 2, 'deflector': 3}
        idx = idx_map.get(weapon_key, -1)
        if idx < 0:
            return False
        try:
            cd = state.weapon_cooldowns[player][idx]
            return int(cd) <= 0
        except (AttributeError, IndexError, TypeError):
            return False

    def _select_build_position(self, state, player):
        positions = self._build_positions.get(player, [])
        if not positions:
            return None
        occupied = set()
        towers = getattr(state, 'towers', [])
        for t in towers:
            if getattr(t, 'player', -1) == player:
                occupied.add((t.x, t.y))
        has_can_apply = hasattr(state, 'can_apply_operation')
        if has_can_apply:
            try:
                from SDK.backend.model import Operation, OperationType
            except ImportError:
                try:
                    from ppo_antwar.sdk.backend.model import Operation, OperationType
                except ImportError:
                    has_can_apply = False
        for pos in positions:
            if pos not in occupied:
                if has_can_apply:
                    op = Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])
                    if not state.can_apply_operation(player, op):
                        continue
                return pos
        return None

    def _select_basic_tower(self, state, player):
        positions = self._build_positions.get(player, [])
        rank = {pos: i for i, pos in enumerate(positions)}
        candidates = [
            t for t in getattr(state, 'towers', [])
            if getattr(t, 'player', -1) == player and getattr(t, 'tower_type', None) == self._TowerType.BASIC
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: rank.get((t.x, t.y), 999))
        return candidates[0]

    def _select_l2_tower(self, state, player):
        positions = self._build_positions.get(player, [])
        rank = {pos: i for i, pos in enumerate(positions)}
        l2_types = {
            self._TowerType.HEAVY, self._TowerType.QUICK,
            self._TowerType.MORTAR, self._TowerType.PRODUCER,
        }
        candidates = [
            t for t in getattr(state, 'towers', [])
            if getattr(t, 'player', -1) == player
            and getattr(t, 'tower_type', None) in l2_types
            and t.tower_id not in self._upgraded_towers
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda t: rank.get((t.x, t.y), 999))
        return candidates[0]

    def _heavies_count(self, state, player):
        count = 0
        for t in getattr(state, 'towers', []):
            if getattr(t, 'player', -1) == player and getattr(t, 'tower_type', None) == self._TowerType.HEAVY:
                count += 1
        return count

    def _try_early_build(self, state, player, coins, tower_count, Operation, OperationType):
        if tower_count > 3:
            return None
        cost = self._get_build_cost(state, tower_count)
        if coins < cost:
            return None
        pos = self._select_build_position(state, player)
        if pos is None:
            return None
        op = Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        return op

    def _try_upgrade_gen_speed_l1(self, state, player, coins, Operation, OperationType):
        if self._get_gen_level(state, player) != 0:
            return None
        if coins < 200:
            return None
        op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        return op

    def _try_upgrade_tower_l2(self, state, player, coins, Operation, OperationType):
        if coins < self._TOWER_L2_COST:
            return None
        tower = self._select_basic_tower(state, player)
        if tower is None:
            return None
        if tower.tower_id in self._upgraded_towers:
            return None
        heavy_count = self._heavies_count(state, player)
        target_type = int(self._TowerType.QUICK) if heavy_count >= 1 else int(self._TowerType.HEAVY)
        op = Operation(OperationType.UPGRADE_TOWER, arg0=tower.tower_id, arg1=target_type)
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        self._upgraded_towers.add(tower.tower_id)
        return op

    def _try_upgrade_gen_speed_l2(self, state, player, coins, Operation, OperationType):
        if self._get_gen_level(state, player) != 1:
            return None
        if coins < 250:
            return None
        op = Operation(OperationType.UPGRADE_GENERATION_SPEED)
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        return op

    def _try_lightning_storm(self, state, player, coins, Operation, OperationType):
        if not self._weapon_ready(state, player, 'lightning_storm'):
            return None
        if coins < self._WEAPON_COST['lightning_storm']:
            return None
        if self._get_round(state) < 50:
            return None
        positions = self._ls_positions.get(player, [(9, 9)])
        for pos in positions:
            op = Operation(OperationType.USE_LIGHTNING_STORM, arg0=pos[0], arg1=pos[1])
            if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
                continue
            return op
        return None

    def _try_emp(self, state, player, coins, Operation, OperationType):
        if not self._weapon_ready(state, player, 'emp_blaster'):
            return None
        if coins < self._WEAPON_COST['emp_blaster']:
            return None
        if self._get_enemy_tower_count(state, player) < 3:
            return None
        positions = self._emp_positions.get(player, [(9, 9)])
        for pos in positions:
            op = Operation(OperationType.USE_EMP_BLASTER, arg0=pos[0], arg1=pos[1])
            if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
                continue
            return op
        return None

    def _try_mid_build(self, state, player, coins, tower_count, Operation, OperationType):
        if tower_count > 6:
            return None
        if tower_count < 4:
            return None
        if coins < 250:
            return None
        cost = self._get_build_cost(state, tower_count)
        if coins < cost:
            return None
        pos = self._select_build_position(state, player)
        if pos is None:
            return None
        op = Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        return op

    def _try_upgrade_ant_hp_l1(self, state, player, coins, Operation, OperationType):
        if self._get_ant_level(state, player) != 0:
            return None
        if coins < 250:
            return None
        op = Operation(OperationType.UPGRADE_GENERATED_ANT)
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        return op

    def _try_upgrade_tower_l3(self, state, player, coins, Operation, OperationType):
        if coins < 300:
            return None
        tower = self._select_l2_tower(state, player)
        if tower is None:
            return None
        if tower.tower_id in self._upgraded_towers:
            return None
        tower_type = getattr(tower, 'tower_type', None)
        if tower_type == self._TowerType.HEAVY:
            target_type = int(self._TowerType.HEAVY_PLUS)
        elif tower_type == self._TowerType.QUICK:
            target_type = int(self._TowerType.SNIPER)
        else:
            return None
        op = Operation(OperationType.UPGRADE_TOWER, arg0=tower.tower_id, arg1=target_type)
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        self._upgraded_towers.add(tower.tower_id)
        return op

    def _try_deflector(self, state, player, coins, Operation, OperationType):
        if not self._weapon_ready(state, player, 'deflector'):
            return None
        if coins < self._WEAPON_COST['deflector']:
            return None
        if self._get_round(state) < 80:
            return None
        positions = self._deflector_positions.get(player, [(9, 9)])
        for pos in positions:
            op = Operation(OperationType.USE_DEFLECTOR, arg0=pos[0], arg1=pos[1])
            if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
                continue
            return op
        return None

    def _try_late_build(self, state, player, coins, tower_count, Operation, OperationType):
        if tower_count < 7:
            return None
        if coins < 350:
            return None
        cost = self._get_build_cost(state, tower_count)
        if coins < cost:
            return None
        pos = self._select_build_position(state, player)
        if pos is None:
            return None
        op = Operation(OperationType.BUILD_TOWER, arg0=pos[0], arg1=pos[1])
        if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
            return None
        return op


class MediumRuleTeacher(MediumRuleAI):
    def __init__(self):
        super().__init__()
        try:
            from ppo_antwar.utils.action_constants import TOWER_POSITIONS
        except ImportError:
            TOWER_POSITIONS = []
        tp_set = set(TOWER_POSITIONS[:10])
        self._build_positions = {
            k: tuple(p for p in v if p in tp_set)
            for k, v in self._build_positions.items()
        }


class PlaceholderAgent(BuiltinAgent):
    def __init__(self, name: str = "Placeholder"):
        self.name = name

    def choose_operations(self, state: Any, player: int) -> List[Any]:
        return []

BUILTIN_AGENTS = {
    'BasicRandomAI': BasicRandomAI,
    'BasicTowerAI': BasicTowerAI,
    'BasicTowAI': BasicTowerAI,
    'MediumRuleAI': MediumRuleAI,
    'MediumRuleTeacher': MediumRuleTeacher,
}


def is_builtin_agent(agent_name: str) -> bool:
    return agent_name in BUILTIN_AGENTS


def create_builtin_agent(agent_name: str, **kwargs) -> Optional[BuiltinAgent]:
    if agent_name not in BUILTIN_AGENTS:
        return None

    try:
        agent_class = BUILTIN_AGENTS[agent_name]
        return agent_class(**kwargs)
    except Exception as e:
        error_context = get_error_context(
            agent_name=agent_name,
            kwargs=kwargs
        )
        log_exception(None, "create_builtin_agent", error_context, e)
        raise  # 根据设计原则，异常应抛向上层


class AgentLoader:
    def __init__(self, logger: BattleLogger = None, path_config: PathConfig = None):
        self.logger = logger
        self.path_config = path_config or PathConfig()

    @staticmethod
    def load_agent_directly(agent_name: str, sdk_path: str, 
                           baseline_path: str) -> Any:
        """
        在子进程中直接加载 agent（进程隔离方式）
        
        Args:
            agent_name: agent 名称
            sdk_path: SDK 路径
            baseline_path: baselines 目录路径（必须由调用方从 PathConfig 获取并传入）
            
        Returns:
            agent 实例或 None
            
        Raises:
            ValueError: 如果 baseline_path 为空
        """
        if not baseline_path:
            raise ValueError(
                "baseline_path is required. "
                "Please get it from PathConfig.baselines_dir in the calling code."
            )
        
        if is_builtin_agent(agent_name):
            return create_builtin_agent(agent_name)

        agent_path = os.path.join(baseline_path, agent_name)
        ai_file = os.path.join(agent_path, 'ai.py')

        if not os.path.exists(agent_path):
            return None
        if not os.path.exists(ai_file):
            return None

        original_sys_path = sys.path.copy()
        original_modules = set(sys.modules.keys())
        original_cwd = os.getcwd()

        modules_to_remove = []
        for key in list(sys.modules.keys()):
            if key in ['ai', 'common', 'SDK', 'AI'] or \
               key.startswith('ai.') or key.startswith('common.') or \
               key.startswith('SDK.') or key.startswith('AI.'):
                modules_to_remove.append(key)

        sdk_parent_path = os.path.dirname(sdk_path)
        sys.path.insert(0, agent_path)
        if sdk_parent_path not in sys.path:
            sys.path.insert(0, sdk_parent_path)

        removed_modules = {}
        for mod in modules_to_remove:
            if mod in sys.modules:
                removed_modules[mod] = sys.modules[mod]
                del sys.modules[mod]

        agent = None
        try:
            os.chdir(agent_path)
            try:
                from ai import create_agent
                agent = create_agent()
            except ImportError:
                from ai import AI
                agent = AI()

            if HAS_TORCH:
                import torch
                target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                for attr_name in ['network', 'neural_agent', 'model', 'nn_model']:
                    if hasattr(agent, attr_name):
                        model_obj = getattr(agent, attr_name)
                        if hasattr(model_obj, 'to'):
                            model_obj.to(target_device)
                            if hasattr(agent, 'device'):
                                agent.device = target_device
                        break

        except Exception as e:
            error_context = get_error_context(
                agent_name=agent_name,
                sdk_path=sdk_path,
                baseline_path=baseline_path,
                agent_path=agent_path,
                sys_path_sample=sys.path[:5]
            )
            log_exception(None, "load_agent_directly", error_context, e)
            raise  # 根据设计原则，异常应抛向上层
        finally:
            os.chdir(original_cwd)
            sys.path = original_sys_path
            
            current_modules = set(sys.modules.keys())
            new_modules = current_modules - original_modules
            
            for mod in new_modules:
                if mod in sys.modules:
                    del sys.modules[mod]
            
            for mod in removed_modules:
                if mod not in sys.modules:
                    sys.modules[mod] = removed_modules[mod]
        
        return agent

    def _info(self, category: str, message: str):
        if self.logger:
            self.logger.info(category, message)

    def _warning(self, category: str, message: str):
        if self.logger:
            self.logger.warning(category, message)

    def _debug(self, category: str, message: str):
        if self.logger:
            self.logger.debug(category, message)

    def _log_agent_load(self, agent_name: str, success: bool, error_msg: str = None):
        if self.logger:
            self.logger.log_agent_load(agent_name, success, error_msg)

    def get_available_agents(self, baseline_path: str = None) -> List[str]:
        available = []

        for agent_name in BUILTIN_AGENTS.keys():
            available.append(agent_name)

        if baseline_path is None:
            baseline_path = str(self.path_config.baselines_dir)

        if os.path.exists(baseline_path):
            for item in os.listdir(baseline_path):
                item_path = os.path.join(baseline_path, item)
                if os.path.isdir(item_path):
                    ai_file = os.path.join(item_path, 'ai.py')
                    if os.path.exists(ai_file):
                        if item not in available:
                            available.append(item)
        else:
            self._warning('load', f"baselines 目录不存在: {baseline_path}")

        return sorted(available)

    def _check_gpu_and_fallback(self, agent: Any) -> Any:
        if not HAS_TORCH:
            return agent
        try:
            has_gpu = torch.cuda.is_available()
            if not has_gpu:
                self._warning('gpu', "GPU 不可用，将模型移至 CPU")
                for attr_name in ['network', 'neural_agent', 'model', 'nn_model']:
                    if hasattr(agent, attr_name):
                        model_obj = getattr(agent, attr_name)
                        if hasattr(model_obj, 'to'):
                            model_obj.to('cpu')
                            if hasattr(agent, 'device'):
                                agent.device = torch.device('cpu')
                            break
        except Exception as e:
            error_context = get_error_context(
                agent_type=type(agent).__name__,
                has_torch=HAS_TORCH
            )
            log_exception(self.logger, "_check_gpu_and_fallback", error_context, e)
        return agent

    def load_agent(self, agent_name: str, baseline_path: str = None) -> Optional[Any]:
        if is_builtin_agent(agent_name):
            agent = create_builtin_agent(agent_name)
            if agent:
                self._log_agent_load(agent_name, True)
                return agent
            self._log_agent_load(agent_name, False, "内置 agent 创建失败")
            return None

        if baseline_path is None:
            baseline_path = str(self.path_config.baselines_dir)

        agent_path = os.path.join(baseline_path, agent_name)

        if not os.path.exists(agent_path):
            self._log_agent_load(agent_name, False, f"目录不存在: {agent_path}")
            return None

        ai_file = os.path.join(agent_path, 'ai.py')
        if not os.path.exists(ai_file):
            self._log_agent_load(agent_name, False, f"缺少 ai.py 文件")
            return None

        # SDK 路径已在 PathConfig 初始化时验证过
        project_sdk_path = str(self.path_config.sdk_dir)

        try:
            agent = self.load_agent_directly(agent_name, project_sdk_path, baseline_path)
            if agent:
                agent = self._check_gpu_and_fallback(agent)
                self._log_agent_load(agent_name, True)
                return agent
            else:
                self._log_agent_load(agent_name, False, "load_agent_directly 返回 None")
                return None

        except Exception as e:
            error_msg = str(e) + "\n" + traceback.format_exc()
            self._log_agent_load(agent_name, False, error_msg)
            return None

    def load_ppo_agent(self, checkpoint_path: str, agent_class=None, **kwargs) -> Optional[Any]:
        if not HAS_TORCH:
            self._log_agent_load('PPO Agent', False, "torch 模块未安装，无法加载 PPO agent")
            return None

        if not os.path.exists(checkpoint_path):
            self._log_agent_load('PPO Agent', False, f"Checkpoint 文件不存在: {checkpoint_path}")
            return None

        try:
            agent = None

            if agent_class is not None:
                agent = agent_class(**kwargs)
                if hasattr(agent, 'load'):
                    agent.load(checkpoint_path)
                elif hasattr(agent, 'load_state_dict'):
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    agent.load_state_dict(state_dict)
            else:
                try:
                    from ppo_antwar.training.policy import PPOPolicy
                    agent = PPOPolicy(**kwargs)
                    agent.load(checkpoint_path)
                except (ImportError, Exception) as e:
                    error_context = get_error_context(
                        checkpoint_path=checkpoint_path,
                        agent_class=agent_class,
                        import_failure=True
                    )
                    log_exception(self.logger, "load_ppo_agent.PPOPolicy_load", error_context, e)

            if agent is None:
                try:
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                    agent = state_dict
                except Exception as e:
                    error_context = get_error_context(
                        checkpoint_path=checkpoint_path,
                        agent_class=agent_class,
                        state_dict_load=True
                    )
                    log_exception(self.logger, "load_ppo_agent.state_dict_load", error_context, e)

            if agent is not None:
                agent = self._check_gpu_and_fallback(agent)
                
                if hasattr(agent, 'eval') and callable(getattr(agent, 'eval')):
                    agent.eval()
                else:
                    for attr_name in ['network', 'model', 'nn_model', 'neural_agent']:
                        if hasattr(agent, attr_name):
                            model_obj = getattr(agent, attr_name)
                            if hasattr(model_obj, 'eval') and callable(getattr(model_obj, 'eval')):
                                model_obj.eval()
                                break
                
                self._log_agent_load('PPO Agent', True)
                return agent
            else:
                self._log_agent_load('PPO Agent', False, "无法加载 PPO agent")
                return None

        except Exception as e:
            error_msg = str(e) + "\n" + traceback.format_exc()
            self._log_agent_load('PPO Agent', False, error_msg)
            return None
