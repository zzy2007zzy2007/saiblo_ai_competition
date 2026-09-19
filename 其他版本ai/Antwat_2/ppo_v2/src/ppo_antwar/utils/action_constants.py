from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Tuple

import torch

MAP_SIZE = 19
EDGE = 10
PLAYER_COUNT = 2

PLAYER_BASES: Tuple[Tuple[int, int], Tuple[int, int]] = (
    (2, EDGE - 1),
    (MAP_SIZE - 3, EDGE - 1),
)


class Terrain(IntEnum):
    VOID = -1
    PATH = 0
    BARRIER = 1
    PLAYER0_HIGHLAND = 2
    PLAYER1_HIGHLAND = 3


MAP_PROPERTY: Tuple[Tuple[int, ...], ...] = (
    (-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 0, -1, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, 0, 0, 1, 0, 1, 0, 0, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, -1, -1, -1, -1),
    (-1, -1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1),
    (0, 0, 2, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0),
    (0, 0, 0, 2, 0, 0, 2, 2, 0, 2, 0, 2, 2, 0, 0, 2, 0, 0, 0),
    (0, 2, 2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 2, 0, 2, 2, 0),
    (0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0),
    (0, 0, 2, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0),
    (0, 1, 3, 0, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 3, 0, 3, 1, 0),
    (0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0),
    (0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 3, 0),
    (0, 3, 0, 0, 0, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 0, 0, 3, 0),
    (0, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 3, 3, 0, 0),
    (-1, 0, 0, 3, 0, 1, 1, 0, 0, 3, 0, 0, 1, 1, 0, 3, 0, 0, -1),
    (-1, -1, -1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, -1, -1, -1),
    (-1, -1, -1, -1, -1, 0, 0, 1, 1, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1),
)

PATH_CELLS: Tuple[Tuple[int, int], ...] = tuple(
    (x, y)
    for x in range(MAP_SIZE)
    for y in range(MAP_SIZE)
    if MAP_PROPERTY[x][y] == Terrain.PATH
)

HIGHLAND_CELLS: Dict[int, List[Tuple[int, int]]] = {
    0: [
        (4, 2),
        (4, 3),
        (4, 9),
        (4, 15),
        (4, 16),
        (5, 3),
        (5, 6),
        (5, 7),
        (5, 9),
        (5, 11),
        (5, 12),
        (5, 15),
        (6, 1),
        (6, 2),
        (6, 4),
        (6, 7),
        (6, 9),
        (6, 11),
        (6, 14),
        (6, 16),
        (6, 17),
        (7, 1),
        (7, 5),
        (7, 8),
        (7, 10),
        (7, 13),
        (7, 17),
        (8, 2),
        (8, 4),
        (8, 7),
        (8, 11),
        (8, 14),
        (8, 16),
    ],
    1: [
        (9, 2),
        (9, 4),
        (9, 14),
        (9, 16),
        (10, 7),
        (10, 8),
        (10, 10),
        (10, 11),
        (11, 1),
        (11, 2),
        (11, 4),
        (11, 5),
        (11, 13),
        (11, 14),
        (11, 16),
        (11, 17),
        (12, 1),
        (12, 6),
        (12, 7),
        (12, 9),
        (12, 11),
        (12, 12),
        (12, 17),
        (13, 2),
        (13, 3),
        (13, 7),
        (13, 9),
        (13, 11),
        (13, 15),
        (13, 16),
        (14, 3),
        (14, 9),
        (14, 15),
    ],
}

VALID_CELLS: Tuple[Tuple[int, int], ...] = (
    PATH_CELLS + tuple(HIGHLAND_CELLS[0]) + tuple(HIGHLAND_CELLS[1]) + PLAYER_BASES
)

ACTION_DIM = 119
MAX_ROUND = 512

# 辅助任务（塔伤害、金币收入预测）的视界数量与步长
NUM_HORIZONS = 5
AUX_HORIZON_STEPS: List[int] = [1, 2, 4, 8, 16]


class SuperWeaponType(IntEnum):
    LIGHTNING_STORM = 1
    EMP_BLASTER = 2
    DEFLECTOR = 3
    EMERGENCY_EVASION = 4


SUPER_WEAPON_POSITIONS: Dict[SuperWeaponType, List[Tuple[int, int]]] = {
    SuperWeaponType.LIGHTNING_STORM: [(7, 9), (8, 9), (9, 9), (10, 9), (11, 9)],
    SuperWeaponType.EMP_BLASTER: [(7, 9), (8, 9), (9, 9), (10, 9), (11, 9)],
    SuperWeaponType.DEFLECTOR: [(4, 9), (5, 9), (6, 9), (12, 9), (13, 9)],
    SuperWeaponType.EMERGENCY_EVASION: [(4, 9), (5, 9), (6, 9), (12, 9), (13, 9)],
}

TOWER_POSITIONS: List[Tuple[int, int]] = [
    (4, 2),
    (4, 9),
    (5, 6),
    (5, 9),
    (6, 7),
    (6, 14),
    (7, 8),
    (8, 7),
    (10, 7),
    (11, 5),
    (11, 14),
    (12, 6),
    (12, 9),
    (13, 9),
    (13, 15),
    (14, 9),
]

TYPE_CONFIG: List[Dict] = [
    {"name": "noop",             "flat_start":   0, "flat_end":   1, "max_targets":   1},
    {"name": "build_tower",      "flat_start":   1, "flat_end":  17, "max_targets":  16},
    {"name": "upgrade_tower",    "flat_start":  17, "flat_end":  81, "max_targets":  64},
    {"name": "downgrade_tower",  "flat_start":  81, "flat_end":  97, "max_targets":  16},
    {"name": "lightning_storm",  "flat_start":  97, "flat_end": 102, "max_targets":   5},
    {"name": "emp_blaster",      "flat_start": 102, "flat_end": 107, "max_targets":   5},
    {"name": "deflector",        "flat_start": 107, "flat_end": 112, "max_targets":   5},
    {"name": "evasion",          "flat_start": 112, "flat_end": 117, "max_targets":   5},
    {"name": "tech_upgrade",     "flat_start": 117, "flat_end": 119, "max_targets":   2},
]

# ── TowerType 合法性校验 ──────────────────────────────────────────────────────
# 从 TOWER_UPGRADE_TREE 推导的完整有效 TowerType 集合
_VALID_TOWER_TYPES: set = {
    0,
    1,
    2,
    3,
    4,
    11,
    12,
    13,
    21,
    22,
    23,
    31,
    32,
    33,
    41,
    42,
    43,
}


def is_valid_tower_type(value) -> bool:
    """检查给定值是否为合法的 TowerType 枚举值。"""
    try:
        return int(value) in _VALID_TOWER_TYPES
    except (ValueError, TypeError):
        return False


def validate_upgrade_operation(op) -> bool:
    """校验 UPGRADE_TOWER 操作的 arg1 是否为有效 TowerType。

    防止构造非法 TowerType 值的操作传入 SDK，触发
        ValueError: X is not a valid TowerType
    """
    from SDK.utils.constants import OperationType

    # 兼容两种 Operation 实现：model.Operation(op_type) 和 forecast.Operation(type)
    op_type = getattr(op, "op_type", None) or getattr(op, "type", None)
    if op_type != OperationType.UPGRADE_TOWER:
        return True
    return is_valid_tower_type(op.arg1)


REWARD_CONFIG: Dict = {
    "base_hp_attack_weight": 2.0,
    "tower_hp_attack_weight": 0.2,
    "own_coin_gain_weight": 0.05,
    "enemy_coin_gain_weight": 0.01,
    "tower_survival_per_tower": 0.02,
    "enemy_tower_survival_per_tower": 0.01,
    "balance_reward_weight": 0.02,
    "own_die_penalty_per_ant": -0.05,
    "tower_survival_level_multipliers": [1.0, 1.5, 2.5],
    "tech_speed_per_level": 1.5,
    "tech_hp_per_level": 1.0,
    "build_tower_tiers": [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1],
    "upgrade_tower_l2": 3.0,
    "upgrade_tower_l3": 5.0,
    "downgrade_tower_penalty": -9.0,
    "upgrade_gen_speed_l1": 15.0,
    "upgrade_gen_speed_l2": 10.0,
    "upgrade_gen_ant_l1": 7.5,
    "upgrade_gen_ant_l2": 2.5,
    "deploy_lightning_storm": 7.5,
    "deploy_emp_blaster": 9.0,
    "deploy_deflector": 1.0,
    "deploy_evasion": 0.8,
    "noop_base_penalty": -0.0025,
    "win_reward": 500.0,
    "loss_reward": -500.0,
    "step_reward_clip": 2000.0,
}

OBS_NORMALIZATION: Dict[str, float] = {
    "hp_scale": 50.0,
    "coin_scale": 1000.0,
    "distance_scale": 19.0,
    "progress_scale": 64.0,
    "tower_count_scale": 20.0,
    "tower_level_scale": 40.0,
    "tower_spread_scale": 10.0,
    "kill_scale": 20.0,
    "tower_spacing_scale": 1.0,
}


# 每个塔位的升级方向数（UPGRADE_TOWER 动作的 target logits 数量）
_UPGRADE_DIRECTIONS = 4

# ── 动作空间一致性验证 ──────────────────────────────────────────────────────
# 运行此函数可确保 TYPE_CONFIG、ACTION_DIM 与 TOWER_POSITIONS 保持一致
# 修改 TOWER_POSITIONS 或 _UPGRADE_DIRECTIONS 后必须重新计算并更新 TYPE_CONFIG


def validate_action_space_consistency():
    """验证动作空间与 TOWER_POSITIONS 一致性"""
    N = len(TOWER_POSITIONS)
    D = _UPGRADE_DIRECTIONS  # = 4

    for cfg in TYPE_CONFIG:
        name = cfg["name"]
        slots = cfg["flat_end"] - cfg["flat_start"]

        if name == "build_tower":
            assert slots == N, f"{name}: {slots} slots != {N} positions"
        elif name == "upgrade_tower":
            assert slots == N * D, f"{name}: {slots} slots != {N}×{D}={N*D}"
        elif name == "downgrade_tower":
            assert slots == N, f"{name}: {slots} slots != {N} positions"

    # 验证 max_targets 与 slot 数一致
    for cfg in TYPE_CONFIG:
        if cfg["name"] == "noop":
            continue
        slots = cfg["flat_end"] - cfg["flat_start"]
        assert cfg["max_targets"] == slots, \
            f"{cfg['name']}: max_targets({cfg['max_targets']}) != slots({slots})"

    # 验证 ACTION_DIM
    total_slots = sum(cfg["flat_end"] - cfg["flat_start"]
                      for cfg in TYPE_CONFIG)
    assert ACTION_DIM == total_slots, \
        f"ACTION_DIM({ACTION_DIM}) != total_slots({total_slots})"

    print(f"[validate] Action space consistent: ACTION_DIM={ACTION_DIM}, TOWER_POSITIONS={N}, UPGRADE_DIRECTIONS={D}")


def action_id_to_type_and_target(
    action_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shape = action_ids.shape
    flat = action_ids.flatten()
    type_ids = torch.zeros_like(flat)
    target_ks = torch.zeros_like(flat)
    for type_id, cfg in enumerate(TYPE_CONFIG):
        in_range = (flat >= cfg["flat_start"]) & (flat < cfg["flat_end"])
        type_ids[in_range] = type_id
        target_ks[in_range] = flat[in_range] - cfg["flat_start"]
    return type_ids.view(shape), target_ks.view(shape)
