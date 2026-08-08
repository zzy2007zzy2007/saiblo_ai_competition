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
    MAP_SIZE,
    PLAYER_BASES,
    SUPER_WEAPON_STATS,
    TOWER_STATS,
    AntBehavior,
    AntKind,
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
        # Probe via clone: apply pending ops first (used_tower / gold context),
        # then the op in question; accepted > 0 means legal.
        c = self._g.clone()
        if pending:
            c.apply_operation_list(player,
                                   [(int(o.op_type), o.arg0, o.arg1) for o in pending])
        _, accepted = c.apply_operation_list(
            player, [(int(operation.op_type), operation.arg0, operation.arg1)])
        return accepted > 0

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

    # ── computed properties (mirror GameState) ──
    def nearest_ant_distance(self, player: int) -> int:
        bx, by = PLAYER_BASES[player]
        enemies = [hex_distance(a.x, a.y, bx, by)
                   for a in self.ants if a.player != player and a.is_alive()]
        return min(enemies) if enemies else 32

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
