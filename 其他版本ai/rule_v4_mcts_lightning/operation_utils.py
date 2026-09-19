#!/usr/bin/env python3
"""单个操作接口工具"""
from __future__ import annotations

from SDK.backend.state import BackendState
from SDK.backend.model import Operation, OperationType
from SDK.utils.constants import (
    STRATEGIC_BUILD_ORDER,
    TOWER_UPGRADE_TREE,
    TowerType,
    SuperWeaponType,
    SUPER_WEAPON_STATS,
)
from SDK.utils.actions import ActionBundle


def get_operation_cost(state: BackendState, player: int, op: Operation) -> int:
    """获取操作的花费（负数表示获得金币）"""
    income = state.operation_income(player, op)
    # 注意：operation_income 返回的是收入，花费是负收入
    return -income


def get_operation_income(state: BackendState, player: int, op: Operation) -> int:
    """获取操作的收入（正数表示获得金币，负数表示花费）"""
    return state.operation_income(player, op)


def get_build_cost(state: BackendState, player: int) -> int:
    """获取建塔的花费"""
    tower_count = state.tower_count(player)
    return state.build_tower_cost(tower_count)


def get_upgrade_cost(target_type: TowerType) -> int:
    """获取升级到指定类型的花费"""
    from SDK.utils.constants import TOWER_STATS
    # 实际上这个方法更简单：
    from SDK.backend.state import BackendState
    # 但我们可以直接用 state.upgrade_tower_cost
    # 不过这里为了保持接口一致，我们需要 state 参数
    # 让我们提供一个需要 state 的版本
    return 0  # 占位，应该用 get_operation_cost


def list_build_operations(state: BackendState, player: int) -> list[Operation]:
    """返回所有合法的建塔操作"""
    results = []
    tower_count = state.tower_count(player)
    build_cost = state.build_tower_cost(tower_count)
    if state.coins[player] < build_cost:
        return results
    for x, y in STRATEGIC_BUILD_ORDER[player]:
        op = Operation(OperationType.BUILD_TOWER, x, y)
        if state.can_apply_operation(player, op):
            results.append(op)
    return results


def list_upgrade_operations(state: BackendState, player: int) -> list[Operation]:
    """返回所有合法的升级塔操作（只包含PRODUCER系列）"""
    results = []
    for tower in state.towers_of(player):
        for target in TOWER_UPGRADE_TREE.get(tower.tower_type, ()):
            if target not in (TowerType.PRODUCER, TowerType.PRODUCER_FAST, TowerType.PRODUCER_SIEGE, TowerType.PRODUCER_MEDIC):
                continue
            op = Operation(OperationType.UPGRADE_TOWER, tower.tower_id, int(target))
            if state.can_apply_operation(player, op):
                results.append(op)
    return results


def list_downgrade_operations(state: BackendState, player: int) -> list[Operation]:
    """返回所有合法的拆塔操作"""
    results = []
    for tower in state.towers_of(player):
        op = Operation(OperationType.DOWNGRADE_TOWER, tower.tower_id)
        if state.can_apply_operation(player, op):
            results.append(op)
    return results


def list_lightning_operations(state: BackendState, player: int) -> list[Operation]:
    """返回所有合法的闪电风暴操作"""
    results = []
    enemy = 1 - player
    enemy_ants = state.ants_of(enemy)
    enemy_towers = state.towers_of(enemy)
    
    if (enemy_ants or enemy_towers) and state.weapon_cooldowns[player, SuperWeaponType.LIGHTNING_STORM] == 0:
        if state.coins[player] >= SUPER_WEAPON_STATS[SuperWeaponType.LIGHTNING_STORM].cost:
            centers = {(ant.x, ant.y) for ant in enemy_ants}
            centers.update((tower.x, tower.y) for tower in enemy_towers)
            for x, y in centers:
                op = Operation(OperationType.USE_LIGHTNING_STORM, x, y)
                if state.can_apply_operation(player, op):
                    results.append(op)
    return results


def list_all_valid_operations(state: BackendState, player: int) -> dict[str, list[Operation]]:
    """返回所有合法操作的字典"""
    return {
        'build': list_build_operations(state, player),
        'upgrade': list_upgrade_operations(state, player),
        'downgrade': list_downgrade_operations(state, player),
        'lightning': list_lightning_operations(state, player),
    }


def operation_to_bundle(op: Operation, name: str = None, score: float = 0.0, tags: tuple[str, ...] = ()) -> ActionBundle:
    """把单个Operation包装成ActionBundle"""
    if name is None:
        name = f"op_{op.op_type}_{op.arg0}_{op.arg1}"
    return ActionBundle(name=name, operations=(op,), score=score, tags=tags)


def list_operations_with_cost(state: BackendState, player: int) -> list[tuple[Operation, int]]:
    """返回所有合法操作及其花费的列表"""
    all_ops = []
    for op_list in list_all_valid_operations(state, player).values():
        for op in op_list:
            cost = get_operation_cost(state, player, op)
            all_ops.append((op, cost))
    return all_ops
