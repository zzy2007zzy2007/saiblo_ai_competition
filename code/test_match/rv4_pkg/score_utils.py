#!/usr/bin/env python3
"""动作score计算工具 - 完全照抄SDK实现"""
from __future__ import annotations

from SDK.backend.state import BackendState
from SDK.backend.model import Operation, OperationType, Tower
from SDK.utils.constants import (
    STRATEGIC_BUILD_ORDER,
    TOWER_UPGRADE_TREE,
    PLAYER_BASES,
    TowerType,
    SuperWeaponType,
    SUPER_WEAPON_STATS,
    LIGHTNING_STORM_ANT_DAMAGE,
    LIGHTNING_STORM_TOWER_DAMAGE,
    LIGHTNING_STORM_TOWER_INTERVAL,
    TOWER_STATS,
)
from SDK.utils.geometry import hex_distance


def calculate_build_score(state: BackendState, player: int, x: int, y: int) -> float:
    """计算建塔操作的score - 完全照抄SDK实现"""
    tower_count = state.tower_count(player)
    build_cost = state.build_tower_cost(tower_count)
    
    pressure = _local_enemy_pressure(state, player, x, y)
    lane_bonus = state.slot_priority(player, x, y)
    score = lane_bonus + pressure * 2.5 - build_cost * 0.03
    
    return score


def calculate_upgrade_score(state: BackendState, player: int, tower_id: int, target_type: TowerType) -> float:
    """计算升级操作的score - 完全照抄SDK实现"""
    tower = state.tower_by_id(tower_id)
    if not tower:
        return 0.0
    
    enemy_base = PLAYER_BASES[1 - player]
    local_density = _local_enemy_pressure(state, player, tower.x, tower.y)
    fit = _tower_type_fit(target_type, local_density, hex_distance(tower.x, tower.y, *enemy_base))
    score = fit + tower.level * 1.5 + state.slot_priority(player, tower.x, tower.y) * 0.15
    
    return score


def calculate_downgrade_score(state: BackendState, player: int, tower_id: int) -> float:
    """计算拆塔操作的score - 完全照抄SDK实现"""
    tower = state.tower_by_id(tower_id)
    if not tower:
        return 0.0
    
    pressure = _local_enemy_pressure(state, player, tower.x, tower.y)
    if pressure > 1.5:
        return -100.0  # 不推荐拆
    
    op = Operation(OperationType.DOWNGRADE_TOWER, tower_id)
    refund = state.operation_income(player, op)
    score = refund * 0.04 - state.slot_priority(player, tower.x, tower.y) * 0.3 - tower.level * 3.0
    
    return score


def calculate_lightning_score(state: BackendState, player: int, x: int, y: int) -> float:
    """计算闪电风暴操作的score - 完全照抄SDK实现"""
    return _storm_value(state, player, x, y)


def calculate_operation_score(state: BackendState, player: int, op: Operation) -> float:
    """计算任意单个操作的score"""
    if op.op_type == OperationType.BUILD_TOWER:
        return calculate_build_score(state, player, op.arg0, op.arg1)
    elif op.op_type == OperationType.UPGRADE_TOWER:
        return calculate_upgrade_score(state, player, op.arg0, TowerType(op.arg1))
    elif op.op_type == OperationType.DOWNGRADE_TOWER:
        return calculate_downgrade_score(state, player, op.arg0)
    elif op.op_type == OperationType.USE_LIGHTNING_STORM:
        return calculate_lightning_score(state, player, op.arg0, op.arg1)
    else:
        return 0.0


# 以下是SDK内部辅助函数的照抄实现

def _local_enemy_pressure(state: BackendState, player: int, x: int, y: int) -> float:
    """照抄SDK的_local_enemy_pressure实现"""
    pressure = 0.0
    for ant in state.ants_of(1 - player):
        distance = hex_distance(x, y, ant.x, ant.y)
        if distance <= 6:
            pressure += max(0.0, 6.5 - distance) * (1.0 + ant.level * 0.4)
    return pressure


def _tower_type_fit(tower_type: TowerType, local_density: float, forward_distance: int) -> float:
    """照抄SDK的_tower_type_fit实现"""
    if tower_type in (TowerType.HEAVY, TowerType.HEAVY_PLUS):
        return local_density * 1.1 - forward_distance * 0.1
    if tower_type == TowerType.ICE:
        return local_density * 1.3
    if tower_type in (TowerType.MORTAR, TowerType.MORTAR_PLUS):
        return local_density * 0.85 + max(0.0, 12 - forward_distance)
    if tower_type in (TowerType.QUICK, TowerType.QUICK_PLUS, TowerType.DOUBLE):
        return local_density * 0.9 + 4.0
    if tower_type == TowerType.SNIPER:
        return max(0.0, 18 - forward_distance) + local_density * 0.4
    if tower_type in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC):
        stats = TOWER_STATS[tower_type]
        cadence = 12.0 / max(stats.spawn_interval, 1)
        density_bonus = max(0.0, 10 - local_density) * 0.75
        forward_bonus = max(0.0, 16 - forward_distance) * 0.22
        branch_bonus = {
            TowerType.PRODUCER: 0.0,
            TowerType.PRODUCER_FAST: 0.5,
            TowerType.PRODUCER_SIEGE: 1.1,
            TowerType.PRODUCER_MEDIC: 1.0,
        }[tower_type]
        return density_bonus + forward_bonus + cadence + branch_bonus
    return 0.0


def _storm_value(state: BackendState, player: int, x: int, y: int) -> float:
    """照抄SDK的_storm_value实现"""
    enemy = 1 - player
    stats = SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM]
    tower_strikes = max(stats.duration // LIGHTNING_STORM_TOWER_INTERVAL, 1)
    total = 0.0
    
    for ant in state.ants_of(enemy):
        distance = hex_distance(x, y, ant.x, ant.y)
        if distance > stats.attack_range:
            continue
        immediate_damage = min(LIGHTNING_STORM_ANT_DAMAGE, ant.hp)
        sustained_damage = min(ant.hp, LIGHTNING_STORM_ANT_DAMAGE * tower_strikes)
        total += immediate_damage / max(ant.max_hp, 1) * (2.5 + ant.level)
        total += sustained_damage / max(ant.max_hp, 1) * (0.8 + ant.level * 0.4)
        if ant.hp <= LIGHTNING_STORM_ANT_DAMAGE:
            total += ant.kill_reward
        total += max(0.0, stats.attack_range + 1 - distance) * 0.2
    
    for tower in state.towers_of(enemy):
        distance = hex_distance(x, y, tower.x, tower.y)
        if distance > stats.attack_range:
            continue
        projected_damage = min(tower.hp, LIGHTNING_STORM_TOWER_DAMAGE * tower_strikes)
        total += projected_damage / max(tower.max_hp, 1) * (6.0 + tower.level * 2.5)
        total += max(0.0, stats.attack_range + 1 - distance) * 0.15
    
    return total - stats.cost * 0.03
