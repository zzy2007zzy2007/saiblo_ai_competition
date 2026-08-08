"""Duck-typed GameState facade over the C++ engine (native_game).

Presents the same interface as ``SDK.backend.engine.GameState`` so that the
bundle MCTS / decoder / FeatureExtractor work UNCHANGED against the C++ engine.
Only the state-creation site needs to change.

Usage:
    from my_ai.az_intent.game_state_facade import GameStateFacade
    state = GameStateFacade.initial(seed=42, cold_handle_rule_illegal=True)
"""
from __future__ import annotations

import numpy as np

from SDK.utils.constants import (
    ANT_KILL_REWARD,
    BASE_UPGRADE_COST,
    CENTERLINE_WEIGHTS,
    COMBAT_ANT_KILL_REWARD,
    LEVEL2_TOWER_UPGRADE_COST,
    LEVEL3_TOWER_UPGRADE_COST,
    MAP_SIZE,
    PLAYER_BASES,
    STRATEGIC_BUILD_ORDER,
    SUPER_WEAPON_STATS,
    TOWER_DOWNGRADE_REFUND_RATIO,
    TOWER_STATS,
    AntBehavior,
    AntKind,
    OperationType,
    SuperWeaponType,
    TowerType,
)
from SDK.utils.geometry import hex_distance


class FacadeBase:
    __slots__ = ("player", "x", "y", "hp", "generation_level", "ant_level")

    def __init__(self, player, x, y, hp, generation_level, ant_level):
        self.player = player
        self.x = x
        self.y = y
        self.hp = hp
        self.generation_level = generation_level
        self.ant_level = ant_level


class FacadeTower:
    __slots__ = ("tower_id", "player", "x", "y", "tower_type",
                 "cooldown_clock", "hp", "level")

    def __init__(self, tower_id, x, y, player, tower_type, hp,
                 hp_limit, level, attack_range, damage, cd):
        self.tower_id = tower_id
        self.player = player
        self.x = x
        self.y = y
        self.tower_type = TowerType(tower_type)
        self.cooldown_clock = float(cd)
        self.hp = hp
        self.level = level

    def stats(self):
        return TOWER_STATS[self.tower_type]

    @property
    def damage(self) -> int:
        return self.stats().damage

    @property
    def speed(self) -> float:
        return self.stats().speed

    @property
    def attack_range(self) -> int:
        return self.stats().attack_range

    @property
    def max_hp(self) -> int:
        return self.stats().max_hp

    @property
    def is_producer(self) -> bool:
        return self.tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST,
                                   TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC)

    def display_cooldown(self) -> int:
        return max(int(self.cooldown_clock), 0)

    def clone(self):
        return self


class FacadeAnt:
    __slots__ = ("ant_id", "player", "x", "y", "hp", "kind", "age",
                 "level", "max_hp", "frozen", "behavior", "status")

    def __init__(self, ant_id, x, y, player, hp, kind, age, level,
                 status, max_hp, frozen, behavior):
        self.ant_id = ant_id
        self.player = player
        self.x = x
        self.y = y
        self.hp = hp
        self.kind = AntKind(kind)
        self.age = age
        self.level = level
        self.status = status
        self.max_hp = max_hp
        self.frozen = bool(frozen)
        self.behavior = AntBehavior(behavior)

    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def kill_reward(self) -> int:
        # Mirrors Ant.kill_reward (COMBAT 18, else ANT_KILL_REWARD[level]).
        if self.kind == AntKind.COMBAT:
            return COMBAT_ANT_KILL_REWARD
        return ANT_KILL_REWARD[self.level]


class FacadeEffect:
    __slots__ = ("weapon_type", "player", "x", "y", "remaining_turns",
                 "last_trigger_round")

    def __init__(self, weapon_type, player, x, y, remaining_turns):
        self.weapon_type = weapon_type
        self.player = player
        self.x = x
        self.y = y
        self.remaining_turns = remaining_turns
        self.last_trigger_round = -1

    def in_range(self, x: int, y: int) -> bool:
        return hex_distance(self.x, self.y, x, y) <= \
            SUPER_WEAPON_STATS[self.weapon_type].attack_range


class GameStateFacade:
    """GameState-compatible view over a native_game.NativeGame.

    Derived views (ants/towers/bases/cooldowns/pheromone/...) are cached per
    state version and invalidated on mutation (apply/advance/clone) — the
    FeatureExtractor re-reads these many times per encode, and rebuilding the
    C++->Python conversion each access was the search bottleneck.
    """

    def __init__(self, native):
        self._g = native
        self._cache: dict = {}

    @classmethod
    def initial(cls, seed: int, cold_handle_rule_illegal: bool = True):
        import native_game
        g = native_game.NativeGame()
        g.init_game(seed=seed, movement_policy="enhanced",
                    cold=cold_handle_rule_illegal)
        return cls(g)

    # ── state operations ──
    def clone(self) -> "GameStateFacade":
        return GameStateFacade(self._g.clone())

    def apply_operation_list(self, player: int, operations) -> None:
        raw = [(int(op.op_type), op.arg0, op.arg1) for op in operations]
        self._g.apply_operation_list(player, raw)
        self._cache.clear()

    def advance_round(self) -> None:
        self._g.advance_round()
        self._cache.clear()

    def can_apply_operation(self, player: int, operation, pending=()) -> bool:
        # Clone-free C++ legality check (dry-run): replicates the cold path's
        # checks with pending's gold/used_tower/camp simulated.  Cheap at any
        # state size (deep-clone per call is ~ms at mid-game — unusable).
        return bool(self._g.can_apply_dry(
            player,
            [int(operation.op_type), operation.arg0, operation.arg1],
            [[int(o.op_type), o.arg0, o.arg1] for o in pending],
        ))

    # ── state properties ──
    @property
    def round_index(self) -> int:
        return self._g.round()

    @property
    def terminal(self) -> bool:
        return self._g.is_ended()

    @property
    def winner(self):
        w = self._g.winner()
        return None if w < 0 else w

    @property
    def bases(self) -> list:
        key = "bases"
        if key not in self._cache:
            out = []
            for p in (0, 1):
                bx, by = PLAYER_BASES[p]
                gen, ant = self._g.base_levels(p)
                out.append(FacadeBase(p, bx, by, self._g.base_hp(p), gen, ant))
            self._cache[key] = out
        return self._cache[key]

    @property
    def coins(self) -> list:
        key = "coins"
        if key not in self._cache:
            self._cache[key] = [self._g.coin(0), self._g.coin(1)]
        return self._cache[key]

    @property
    def ants(self) -> list:
        key = "ants"
        if key not in self._cache:
            self._cache[key] = [FacadeAnt(*row) for row in self._g.ant_details()]
        return self._cache[key]

    @property
    def towers(self) -> list:
        key = "towers"
        if key not in self._cache:
            self._cache[key] = [FacadeTower(*row) for row in self._g.tower_details()]
        return self._cache[key]

    @property
    def weapon_cooldowns(self):
        key = "weapon_cd"
        if key not in self._cache:
            arr = np.zeros((2, 5), dtype=np.int16)  # columns 1-4 = SuperWeaponType
            for p in (0, 1):
                for i, cd in enumerate(self._g.weapon_cds(p)):
                    arr[p, i + 1] = cd
            self._cache[key] = arr
        return self._cache[key]

    @property
    def super_weapon_usage(self) -> list:
        key = "usage"
        if key not in self._cache:
            self._cache[key] = [self._g.super_weapon_usage(0),
                                self._g.super_weapon_usage(1)]
        return self._cache[key]

    @property
    def pheromone(self):
        key = "pheromone"
        if key not in self._cache:
            self._cache[key] = np.array(self._g.pheromone_flat(),
                                        dtype=np.float64).reshape(2, MAP_SIZE, MAP_SIZE)
        return self._cache[key]

    @property
    def old_count(self) -> list:  # C++ does not track these; stub
        return [0, 0]

    @property
    def die_count(self) -> list:
        return [0, 0]

    @property
    def active_effects(self) -> list:
        key = "effects"
        if key not in self._cache:
            self._cache[key] = [FacadeEffect(SuperWeaponType(t + 1), p, x, y, d)
                                for t, p, x, y, d in self._g.active_effects()]
        return self._cache[key]

    # ── query helpers ──
    def tower_at(self, x: int, y: int):
        for t in self.towers:
            if t.x == x and t.y == y:
                return t
        return None

    def towers_of(self, player: int) -> list:
        return [t for t in self.towers if t.player == player]

    def ants_of(self, player: int) -> list:
        return [a for a in self.ants if a.player == player]

    def tower_count(self, player: int) -> int:
        return len(self.towers_of(player))

    def build_tower_cost(self, tower_count: int) -> int:
        return type(self._g).tower_build_cost(tower_count)

    def tower_by_id(self, tower_id: int):
        for tower in self.towers:
            if tower.tower_id == tower_id:
                return tower
        return None

    def strategic_slots(self, player: int) -> tuple:
        return STRATEGIC_BUILD_ORDER[player]

    def upgrade_tower_cost(self, target_type) -> int:
        # Basic/level-2 towers cost 60, upgraded level-3 cost 200 — mirrors
        # GameState.upgrade_tower_cost (and the C++ tower_upgrade_cost helper).
        if target_type.value < 10:
            return LEVEL2_TOWER_UPGRADE_COST
        return LEVEL3_TOWER_UPGRADE_COST

    def destroy_tower_income(self, tower_count: int, tower=None) -> int:
        refund = self.build_tower_cost(tower_count - 1) * TOWER_DOWNGRADE_REFUND_RATIO
        if tower is None:
            return int(refund)
        return int(refund * max(tower.hp, 0) / max(tower.max_hp, 1))

    def downgrade_tower_income(self, tower_type, tower=None) -> int:
        refund = self.upgrade_tower_cost(tower_type) * TOWER_DOWNGRADE_REFUND_RATIO
        if tower is None:
            return int(refund)
        return int(refund * max(tower.hp, 0) / max(tower.max_hp, 1))

    def upgrade_base_cost(self, level: int) -> int:
        return BASE_UPGRADE_COST[level]

    def weapon_cost(self, weapon_type) -> int:
        return SUPER_WEAPON_STATS[weapon_type].cost

    def operation_income(self, player: int, operation, tower_count_hint=None) -> int:
        # Mirrors GameState._operation_income (net gold change of the op).
        op_type = operation.op_type
        if op_type == OperationType.BUILD_TOWER:
            count = self.tower_count(player) if tower_count_hint is None else tower_count_hint
            return -self.build_tower_cost(count)
        if op_type == OperationType.UPGRADE_TOWER:
            return -self.upgrade_tower_cost(TowerType(operation.arg1))
        if op_type == OperationType.DOWNGRADE_TOWER:
            tower = self.tower_by_id(operation.arg0)
            if tower is None:
                return 0
            if tower.tower_type == TowerType.BASIC:
                count = self.tower_count(player) if tower_count_hint is None else tower_count_hint
                return self.destroy_tower_income(count, tower)
            return self.downgrade_tower_income(tower.tower_type, tower)
        if op_type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER,
                       OperationType.USE_DEFLECTOR, OperationType.USE_EMERGENCY_EVASION):
            return -self.weapon_cost(SuperWeaponType(op_type % 10))
        if op_type == OperationType.UPGRADE_GENERATION_SPEED:
            level = self.bases[player].generation_level
            return -self.upgrade_base_cost(level) if level < len(BASE_UPGRADE_COST) else 0
        if op_type == OperationType.UPGRADE_GENERATED_ANT:
            level = self.bases[player].ant_level
            return -self.upgrade_base_cost(level) if level < len(BASE_UPGRADE_COST) else 0
        return 0

    # ── computed properties (mirror GameState) ──
    def nearest_ant_distance(self, player: int) -> int:
        bx, by = PLAYER_BASES[player]
        enemies = [hex_distance(a.x, a.y, bx, by)
                   for a in self.ants if a.player != player and a.is_alive()]
        return min(enemies) if enemies else 32

    def frontline_distance(self, player: int) -> int:
        # Mirrors GameState.frontline_distance: nearest of player's OWN ants to
        # the ENEMY base (32 when none alive).
        bx, by = PLAYER_BASES[1 - player]
        own = [hex_distance(a.x, a.y, bx, by)
               for a in self.ants if a.player == player and a.is_alive()]
        return min(own) if own else 32

    def safe_coin_threshold(self, player: int) -> int:
        enemy = 1 - player
        emp_stats = SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER]
        emp_cd = int(self.weapon_cooldowns[enemy, SuperWeaponType.EMP_BLASTER])
        enemy_coin = int(self.coins[enemy])
        capped_cost = max(emp_stats.cost - 1, 0)
        if emp_cd >= emp_stats.cooldown - 10:
            return 0
        if emp_cd > 0:
            return max(int(min(enemy_coin, capped_cost) - emp_cd * 1.66), 0)
        return min(enemy_coin, capped_cost)

    def tower_spread_score(self, player: int) -> float:
        towers = self.towers_of(player)
        if len(towers) < 2:
            return 0.0
        penalty = 0.0
        for i, tower in enumerate(towers[:-1]):
            for other in towers[i + 1:]:
                d = hex_distance(tower.x, tower.y, other.x, other.y)
                if d <= 3:
                    penalty += 5.0
                elif d <= 6:
                    penalty += 2.0
        return -penalty

    def slot_priority(self, player: int, x: int, y: int) -> float:
        # Pure heuristic (STRATEGIC_BUILD_ORDER / CENTERLINE_WEIGHTS), mirrors
        # GameState.slot_priority — no engine state involved.
        try:
            order = self.strategic_slots(player).index((x, y))
        except ValueError:
            order = len(self.strategic_slots(player))
        priority = max(0.0, 24.0 - order * 0.6)
        priority *= CENTERLINE_WEIGHTS.get((x, y), 1.0)
        base_x, base_y = PLAYER_BASES[player]
        priority += hex_distance(x, y, base_x, base_y) * 0.4
        return priority
