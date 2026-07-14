"""Score regression: per-cell decoder + ActionCatalog scoring functions."""

from __future__ import annotations

import numpy as np

from SDK.utils.constants import (
    HIGHLAND_CELLS, MAP_SIZE, PLAYER_BASES,
    TOWER_UPGRADE_TREE, SUPER_WEAPON_STATS, SuperWeaponType,
    TowerType, OperationType,
    ANT_MAX_HP, ANT_GENERATION_CYCLE,
    LEVEL2_TOWER_UPGRADE_COST, LEVEL3_TOWER_UPGRADE_COST,
    LIGHTNING_STORM_ANT_DAMAGE, LIGHTNING_STORM_TOWER_DAMAGE,
    LIGHTNING_STORM_TOWER_INTERVAL, MAX_ROUND,
)
from SDK.backend.model import Operation
from SDK.backend.state import BackendState
from SDK.utils.geometry import hex_distance
from SDK.utils.constants import TOWER_STATS  # for _tower_type_fit
from my_ai.decoder import (
    CHANNEL_TO_TOWER_TYPE, CHANNEL_TO_SUPER_WEAPON,
    SUPER_WEAPON_TO_OP_TYPE, upgrade_step, make_position_masks,
)

NUM_CLASSES = 24
MAP_SIZE = 19


# ═══════════════════════════════════════════════════════════════
# 1. Per-cell decoder
# ═══════════════════════════════════════════════════════════════

def decode_single_cell(
    class_id: int, x: int, y: int, state: BackendState, player: int,
) -> Operation | None:
    """Decode a single (class_id, x, y) into an Operation, or None if illegal/HOLD.

    Follows the same logic as decoder.py's decode_head, but for a specific
    position instead of argmax over action_map.
    """
    pos_mask = make_position_masks(state, player)
    if not pos_mask[class_id, x, y]:
        return None  # illegal position for this class

    # HOLD
    if class_id == 23:
        return None

    # Base upgrades (no position)
    if class_id == 21:
        return Operation(OperationType.UPGRADE_GENERATION_SPEED)
    if class_id == 22:
        return Operation(OperationType.UPGRADE_GENERATED_ANT)

    # Super weapons (17-20)
    if 17 <= class_id <= 20:
        sw_type = CHANNEL_TO_SUPER_WEAPON[class_id]
        op_type = SUPER_WEAPON_TO_OP_TYPE[sw_type]
        return Operation(op_type, x, y)

    # Tower actions (0-15)
    if 0 <= class_id <= 15:
        return _decode_tower_cell(state, player, class_id, x, y)

    # Downgrade (16)
    if class_id == 16:
        tower = state.tower_at(x, y)
        if tower is not None and tower.player == player:
            return Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)

    return None


def _decode_tower_cell(
    state: BackendState, player: int, class_id: int, x: int, y: int,
) -> Operation | None:
    """Decode a tower action (build or upgrade toward target) at (x,y)."""
    target_type = CHANNEL_TO_TOWER_TYPE[class_id]
    tower = state.tower_at(x, y)

    if tower is None:
        # Empty highland: build
        return Operation(OperationType.BUILD_TOWER, x, y)

    if tower.player != player:
        return None  # enemy tower

    step = upgrade_step(tower.tower_type, target_type)
    if step is None:
        return None  # already at target

    return Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(step))


# ═══════════════════════════════════════════════════════════════
# 2. Single-operation scoring (replicates ActionCatalog formulas)
# ═══════════════════════════════════════════════════════════════

def score_operation(op: Operation, state: BackendState, player: int) -> float:
    """Score a single Operation using ActionCatalog-inspired heuristics.

    Returns a scalar score (higher = better). Not normalized.
    """
    t = op.op_type
    if t == OperationType.BUILD_TOWER:
        return _score_build(state, player, op.arg0, op.arg1)
    elif t == OperationType.UPGRADE_TOWER:
        return _score_upgrade(state, player, op.arg0, op.arg1)
    elif t == OperationType.DOWNGRADE_TOWER:
        return _score_downgrade(state, player, op.arg0)
    elif t == OperationType.UPGRADE_GENERATION_SPEED:
        return _upgrade_gen_value(state, player)
    elif t == OperationType.UPGRADE_GENERATED_ANT:
        return _upgrade_ant_value(state, player)
    elif t == OperationType.USE_LIGHTNING_STORM:
        return _storm_value(state, player, op.arg0, op.arg1)
    elif t == OperationType.USE_EMP_BLASTER:
        return _score_emp(state, player, op.arg0, op.arg1)
    elif t == OperationType.USE_DEFLECTOR:
        return _score_deflector(state, player, op.arg0, op.arg1)
    elif t == OperationType.USE_EMERGENCY_EVASION:
        return _score_evasion(state, player, op.arg0, op.arg1)
    return 0.0


# ── Scoring sub-functions (matching ActionCatalog logic) ──────

def _local_enemy_pressure(state, player, x, y):
    """Nearby enemy ant threat at (x,y)."""
    pressure = 0.0
    for ant in state.ants_of(1 - player):
        d = hex_distance(x, y, ant.x, ant.y)
        if d <= 6:
            pressure += max(0.0, 6.5 - d) * (1.0 + ant.level * 0.4)
    return pressure


def _tower_type_fit(tower_type, local_density, forward_distance):
    """How well a tower type fits the current situation."""
    if tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS, TowerType.BEWITCH):
        return local_density * 1.1 - forward_distance * 0.1
    if tower_type in (TowerType.ICE, TowerType.PULSE):
        return local_density * 1.3
    if tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS, TowerType.MISSILE):
        return local_density * 0.85 + max(0.0, 12 - forward_distance)
    if tower_type in (TowerType.QUICK, TowerType.QUICK_PLUS, TowerType.DOUBLE):
        return local_density * 0.9 + 4.0
    if tower_type == TowerType.SNIPER:
        return max(0.0, 18 - forward_distance) + local_density * 0.4
    if tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST,
                      TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC):
        stats = TOWER_STATS[tower_type]
        cadence = 12.0 / max(stats.spawn_interval, 1)
        density_bonus = max(0.0, 10 - local_density) * 0.75
        forward_bonus = max(0.0, 16 - forward_distance) * 0.22
        branch_bonus = {TowerType.PRODUCER: 0.0, TowerType.PRODUCER_FAST: 0.5,
                        TowerType.PRODUCER_SIEGE: 1.1, TowerType.PRODUCER_MEDIC: 1.0}[tower_type]
        return density_bonus + forward_bonus + cadence + branch_bonus
    return 0.0


def _score_build(state, player, x, y):
    pressure = _local_enemy_pressure(state, player, x, y)
    lane_bonus = state.slot_priority(player, x, y)
    build_cost = state.build_tower_cost(state.tower_count(player))
    return lane_bonus + pressure * 2.5 - build_cost * 0.03


def _score_upgrade(state, player, tower_id, target_type):
    # Find the tower
    tower = None
    for t in state.towers_of(player):
        if t.tower_id == tower_id:
            tower = t
            break
    if tower is None:
        return 0.0

    pressure = _local_enemy_pressure(state, player, tower.x, tower.y)
    enemy_base = PLAYER_BASES[1 - player]
    forward_distance = hex_distance(tower.x, tower.y, enemy_base[0], enemy_base[1])
    fit = _tower_type_fit(target_type, pressure, forward_distance)
    return fit + tower.level * 1.5 + state.slot_priority(player, tower.x, tower.y) * 0.15


def _score_downgrade(state, player, tower_id):
    for t in state.towers_of(player):
        if t.tower_id == tower_id:
            pressure = _local_enemy_pressure(state, player, t.x, t.y)
            if pressure > 1.5:
                return -100.0  # strong penalty: don't downgrade under pressure
            refund = state.operation_income(player, Operation(OperationType.DOWNGRADE_TOWER, tower_id))
            return refund * 0.04 - state.slot_priority(player, t.x, t.y) * 0.3 - t.level * 3.0
    return 0.0


def _upgrade_gen_value(state, player):
    level = state.bases[player].generation_level
    current_cycle = ANT_GENERATION_CYCLE[level]
    next_cycle = ANT_GENERATION_CYCLE[level + 1]
    if next_cycle >= current_cycle - 1e-6:
        return 0.0
    tempo_gain = current_cycle - next_cycle
    return (10.0 + tempo_gain * 14.0
            + state.nearest_ant_distance(player) * 0.12
            - state.round_index * 0.015)


def _upgrade_ant_value(state, player):
    level = state.bases[player].ant_level
    hp_gain = ANT_MAX_HP[level + 1] - ANT_MAX_HP[level]
    if hp_gain <= 0:
        return 0.0
    return (8.0 + hp_gain * 1.4
            + state.frontline_distance(player) * 0.22
            - state.round_index * 0.01
            - level * 1.2)


def _storm_value(state, player, x, y):
    enemy = 1 - player
    stats = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM]
    tower_strikes = max(stats.duration // LIGHTNING_STORM_TOWER_INTERVAL, 1)
    total = 0.0
    for ant in state.ants_of(enemy):
        d = hex_distance(x, y, ant.x, ant.y)
        if d > stats.attack_range:
            continue
        immediate = min(LIGHTNING_STORM_ANT_DAMAGE, ant.hp)
        sustained = min(ant.hp, LIGHTNING_STORM_ANT_DAMAGE * tower_strikes)
        total += immediate / max(ant.max_hp, 1) * (2.5 + ant.level)
        total += sustained / max(ant.max_hp, 1) * (0.8 + ant.level * 0.4)
        if ant.hp <= LIGHTNING_STORM_ANT_DAMAGE:
            total += ant.kill_reward
        total += max(0.0, stats.attack_range + 1 - d) * 0.2
    for tower in state.towers_of(enemy):
        d = hex_distance(x, y, tower.x, tower.y)
        if d > stats.attack_range:
            continue
        projected = min(tower.hp, LIGHTNING_STORM_TOWER_DAMAGE * tower_strikes)
        total += projected / max(tower.max_hp, 1) * (6.0 + tower.level * 2.5)
        total += max(0.0, stats.attack_range + 1 - d) * 0.15
    return total - stats.cost * 0.03


def _score_emp(state, player, x, y):
    total = 0.0
    for tower in state.towers_of(1 - player):
        d = hex_distance(x, y, tower.x, tower.y)
        if d <= SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].attack_range:
            total += 3.0 + tower.level * 2.5
    return total - SUPER_WEAPON_STATS[SuperWeaponType.EMP_BLASTER].cost * 0.025


def _score_deflector(state, player, x, y):
    total = 0.0
    for ant in state.ants_of(player):
        if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].attack_range:
            total += 0.8 + ant.level * 0.8
    total += max(0.0, 7 - state.nearest_ant_distance(player)) * 0.5
    return total - SUPER_WEAPON_STATS[SuperWeaponType.DEFLECTOR].cost * 0.02


def _score_evasion(state, player, x, y):
    total = 0.0
    for ant in state.ants_of(player):
        if hex_distance(x, y, ant.x, ant.y) <= SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].attack_range:
            total += 0.6 + ant.level * 0.7
    total += max(0.0, 5 - state.nearest_ant_distance(player))
    return total - SUPER_WEAPON_STATS[SuperWeaponType.EMERGENCY_EVASION].cost * 0.02


# ═══════════════════════════════════════════════════════════════
# 3. Build full score map for a game state
# ═══════════════════════════════════════════════════════════════

def build_score_map(
    state: BackendState, player: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute full (24, 19, 19) score_map and (24,) class_scores for a state.

    For each legal (class_id, x, y), decodes the action and scores it.
    Returns:
        score_map: (24, 19, 19) float32 — ActionCatalog score per cell
        class_scores: (24,) float32 — max score per class
    """
    pos_mask = make_position_masks(state, player)
    score_map = np.zeros((NUM_CLASSES, MAP_SIZE, MAP_SIZE), dtype=np.float32)

    for class_id in range(NUM_CLASSES):
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                if not pos_mask[class_id, x, y]:
                    continue
                op = decode_single_cell(class_id, x, y, state, player)
                if op is not None:
                    score_map[class_id, x, y] = score_operation(op, state, player)

    class_scores = score_map.reshape(NUM_CLASSES, -1).max(axis=1)
    return score_map, class_scores
