from typing import Any, List, Optional
from abc import ABC, abstractmethod


class BuiltinAgent(ABC):
    @abstractmethod
    def choose_operations(self, state: Any, player: int) -> List[Any]:
        raise NotImplementedError

    def choose_bundle(self, state: Any, player: int, bundles: Optional[List] = None):
        operations = self.choose_operations(state, player)
        return operations


class MediumRuleAI(BuiltinAgent):

    _WEAPON_COST = {
        'lightning_storm': 90,
        'emp_blaster': 135,
        'deflector': 60,
    }

    _TOWER_L2_COST = 60
    _TOWER_L3_COST = 200

    def __init__(self):
        try:
            from SDK.utils.constants import STRATEGIC_BUILD_ORDER, TowerType, SuperWeaponType
        except ImportError:
            STRATEGIC_BUILD_ORDER = {0: [], 1: []}
            TowerType = None
            SuperWeaponType = None

        self._build_positions = STRATEGIC_BUILD_ORDER
        self._TowerType = TowerType
        self._SuperWeaponType = SuperWeaponType
        self._upgraded_towers: set = set()
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
        from SDK.backend.model import Operation, OperationType

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

        except Exception:
            return []

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
        return self._get_tower_count(state, 1 - player)

    def _get_round(self, state):
        return getattr(state, 'round_index', 0)

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
        cost = 15 * (3 ** (tower_count // 2))
        if tower_count % 2 == 1:
            cost *= 2
        return cost

    def _weapon_ready(self, state, player, weapon_key):
        idx_map = {'lightning_storm': 1, 'emp_blaster': 2, 'deflector': 3}
        idx = idx_map.get(weapon_key, -1)
        if idx < 0:
            return False
        try:
            return int(state.weapon_cooldowns[player][idx]) <= 0
        except (AttributeError, IndexError, TypeError):
            return False

    def _select_build_position(self, state, player):
        positions = self._build_positions.get(player, [])
        if not positions:
            return None
        occupied = {
            (t.x, t.y) for t in getattr(state, 'towers', [])
            if getattr(t, 'player', -1) == player
        }
        has_can_apply = hasattr(state, 'can_apply_operation')
        if has_can_apply:
            try:
                from SDK.backend.model import Operation, OperationType
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
            if getattr(t, 'player', -1) == player
            and getattr(t, 'tower_type', None) == self._TowerType.BASIC
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
        return sum(
            1 for t in getattr(state, 'towers', [])
            if getattr(t, 'player', -1) == player
            and getattr(t, 'tower_type', None) == self._TowerType.HEAVY
        )

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
        if tower is None or tower.tower_id in self._upgraded_towers:
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
        for pos in self._ls_positions.get(player, [(9, 9)]):
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
        for pos in self._emp_positions.get(player, [(9, 9)]):
            op = Operation(OperationType.USE_EMP_BLASTER, arg0=pos[0], arg1=pos[1])
            if hasattr(state, 'can_apply_operation') and not state.can_apply_operation(player, op):
                continue
            return op
        return None

    def _try_mid_build(self, state, player, coins, tower_count, Operation, OperationType):
        if tower_count > 6 or tower_count < 4:
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
        if tower is None or tower.tower_id in self._upgraded_towers:
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
        for pos in self._deflector_positions.get(player, [(9, 9)]):
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
