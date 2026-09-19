from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Tuple

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

MAX_ROUND = 512

# ── 动作空间维度 ──────────────────────────────────────────────────────────────
# 27 = 3 (strategy) + 6 (type) + 10 (sub_type) + 8 (position)
ACTION_DIM = 27

# ── 语义分段定义 ──────────────────────────────────────────────────────────────
ACTION_SEGMENTS: Dict[str, Tuple[int, int]] = {
    "strategy":  (0, 3),    # [0:3)   strategy logits → {0,1,2} [防御/无操作/进攻]
    "type":      (3, 9),    # [3:9)   type_idx logits  → {0..5}
    "sub_type":  (9, 19),   # [9:19)  sub_type logits  → {0..9}
    "position":  (19, 27),  # [19:27) position logits  → {0..7}
}

# sub_type 段进一步分解为三个子段（仅对 UPGRADE_TOWER 有意义）
SUB_TYPE_SEGMENTS: Dict[str, Tuple[int, int]] = {
    "main_tree":    (9, 12),    # [9:12)  HEAVY/QUICK/MORTAR
    "sub_variant":  (12, 15),   # [12:15) variant_A/variant_B/variant_C
    "producer":     (15, 19),   # [15:19) PRODUCER/FAST/SIEGE/MEDIC
}


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


# ── 多表位置映射常量 ──────────────────────────────────────────────────────────

# 塔操作：position 0-7 → 当前玩家的固定塔位（按战略意义排序，双方镜像对称）
_PLAYER_TOWER_POS: Dict[int, List[Tuple[int, int]]] = {
    0: [  # Player 0
        (5, 9), (4, 9), (7, 8), (6, 7),   # pos 0-3: 中线核心 → 中央战场
        (8, 7), (5, 6), (6, 14), (4, 2),   # pos 4-7: 中央侧翼 → 角落防守
    ],
    1: [  # Player 1（镜像对称）
        (13, 9), (14, 9), (10, 7), (12, 6),  # pos 0-3
        (11, 5), (12, 9), (11, 14), (13, 15), # pos 4-7
    ],
}

# 超级武器：position 0-7 → 5 个预设位的冗余映射
# pos 5-7 冗余映射到 pos 2-4（中线/核心位），提高合法率
_WEAPON_POS_REDUNDANCY: List[int] = [0, 1, 2, 3, 4, 2, 3, 4]

# 科技升级：position 0-7 → 2 种科技类型的冗余映射
_TECH_POS_MAP: Dict[int, str] = {
    0: "generation_speed",
    1: "generation_speed",
    2: "generation_speed",
    3: "generation_speed",
    4: "generated_ant",
    5: "generated_ant",
    6: "generated_ant",
    7: "generated_ant",
}


# ── (Strategy, Type) 联合映射表 ───────────────────────────────────────────────
# strategy=0 防御, strategy=1 无操作, strategy=2 进攻
# 每个条目: (action_name, flags)
#   flags = {"use_sub_type": True} 表示该操作需要解析 sub_type
#   None 表示 → NOOP
ACTION_DECODE_TABLE: List[Dict[int, Tuple[str, Dict] | None]] = [
    # strategy=0 (防御)
    {
        0: ("build_tower",          {}),
        1: ("upgrade_tower",        {"use_sub_type": True}),
        2: ("lightning_storm",      {}),  # 杀伤敌方蚂蚁
        3: ("lightning_storm",      {}),  # 冗余
        4: ("lightning_storm",      {}),  # 冗余
        5: None,                          # 进攻独占 slot → NOOP
    },
    # strategy=1 (无操作) — 全部 NOOP
    {i: None for i in range(6)},
    # strategy=2 (进攻)
    {
        0: ("downgrade_tower",      {}),
        1: ("downgrade_tower",      {}),  # 冗余
        2: ("deflector",            {}),  # 保护己方蚂蚁
        3: ("evasion",              {}),  # 己方蚂蚁闪避
        4: ("emp_blaster",          {}),  # 压制敌方塔
        5: ("tech_upgrade",         {}),
    },
]


# ── TowerType 合法性校验 ──────────────────────────────────────────────────────
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
