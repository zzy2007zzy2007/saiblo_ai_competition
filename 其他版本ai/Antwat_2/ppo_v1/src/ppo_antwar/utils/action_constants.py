from __future__ import annotations
from enum import IntEnum
from typing import Dict, List, Tuple

MAP_SIZE = 19
EDGE = 10
PLAYER_COUNT = 2
PLAYER_BASES = ((2, EDGE - 1), (MAP_SIZE - 3, EDGE - 1))

class Terrain(IntEnum):
    VOID = -1
    PATH = 0
    BARRIER = 1
    PLAYER0_HIGHLAND = 2
    PLAYER1_HIGHLAND = 3

MAP_PROPERTY = (
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

PATH_CELLS = tuple(
    (x, y)
    for x in range(MAP_SIZE)
    for y in range(MAP_SIZE)
    if MAP_PROPERTY[x][y] == Terrain.PATH
)

HIGHLAND_CELLS: Dict[int, List[Tuple[int, int]]] = {
    0: [(4, 2), (4, 3), (4, 9), (4, 15), (4, 16), (5, 3), (5, 6), (5, 7), (5, 9), (5, 11), (5, 12), (5, 15), (6, 1), (6, 2), (6, 4), (6, 7), (6, 9), (6, 11), (6, 14), (6, 16), (6, 17), (7, 1), (7, 5), (7, 8), (7, 10), (7, 13), (7, 17), (8, 2), (8, 4), (8, 7), (8, 11), (8, 14), (8, 16)],
    1: [(9, 2), (9, 4), (9, 14), (9, 16), (10, 7), (10, 8), (10, 10), (10, 11), (11, 1), (11, 2), (11, 4), (11, 5), (11, 13), (11, 14), (11, 16), (11, 17), (12, 1), (12, 6), (12, 7), (12, 9), (12, 11), (12, 12), (12, 17), (13, 2), (13, 3), (13, 7), (13, 9), (13, 11), (13, 15), (13, 16), (14, 3), (14, 9), (14, 15)],
}

VALID_CELLS = PATH_CELLS + tuple(HIGHLAND_CELLS[0]) + tuple(HIGHLAND_CELLS[1]) + PLAYER_BASES

MAX_ACTIONS = 96
ACTION_DIM = 96


class SuperWeaponType(IntEnum):
    LIGHTNING_STORM = 1
    EMP_BLASTER = 2
    DEFLECTOR = 3
    EMERGENCY_EVASION = 4


class OperationType(IntEnum):
    BUILD_TOWER = 11
    UPGRADE_TOWER = 12
    DOWNGRADE_TOWER = 13
    USE_LIGHTNING_STORM = 21
    USE_EMP_BLASTER = 22
    USE_DEFLECTOR = 23
    USE_EMERGENCY_EVASION = 24
    UPGRADE_GENERATION_SPEED = 31
    UPGRADE_GENERATED_ANT = 32
    NO_OP = 0


class AntBehavior(IntEnum):
    IDLE = 0
    RANDOM = 1
    BEWITCHED = 2
    CONTROL_FREE = 3


SUPER_WEAPON_POSITIONS: Dict[SuperWeaponType, List[Tuple[int, int]]] = {
    SuperWeaponType.LIGHTNING_STORM: [(7, 9), (8, 9), (9, 9), (10, 9), (11, 9)],
    SuperWeaponType.EMP_BLASTER: [(7, 9), (8, 9), (9, 9), (10, 9), (11, 9)],
    SuperWeaponType.DEFLECTOR: [(4, 9), (5, 9), (6, 9), (12, 9), (13, 9)],
    SuperWeaponType.EMERGENCY_EVASION: [(4, 9), (5, 9), (6, 9), (12, 9), (13, 9)],
}

TOWER_POSITIONS: List[Tuple[int, int]] = [
    (4, 2), (4, 9), (5, 6), (5, 9), (6, 7), (6, 14), (7, 8), (8, 7),
    (10, 7), (11, 5), (11, 14), (12, 6), (12, 9), (13, 9), (13, 15), (14, 9),
]

ACTION_OFFSETS: Dict[str, int] = {
    'build_tower': 1,
    'upgrade_tower': 11,
    'downgrade_tower': 51,
    'lightning_storm': 61,
    'emp_blaster': 66,
    'deflector': 71,
    'evasion': 76,
    'tech_upgrade': 81,
}

ACTION_SPACE_CONFIG: Dict[str, dict] = {
    'build_tower': {'start': 1, 'end': 11},
    'upgrade_tower': {'start': 11, 'end': 51},
    'downgrade_tower': {'start': 51, 'end': 61},
    'use_lightning_storm': {'start': 61, 'end': 66},
    'use_emp_blaster': {'start': 66, 'end': 71},
    'use_deflector': {'start': 71, 'end': 76},
    'use_emergency_evasion': {'start': 76, 'end': 81},
    'upgrade_generation_speed': {'start': 81, 'end': 82},
    'upgrade_generated_ant': {'start': 82, 'end': 83},
}

TYPE_CONFIG: List[Dict] = [
    {'name': 'noop',              'flat_start': 0,   'flat_end': 1,    'max_targets': 1},
    {'name': 'build_tower',       'flat_start': 1,   'flat_end': 11,   'max_targets': 16},
    {'name': 'upgrade_tower',     'flat_start': 11,  'flat_end': 51,   'max_targets': 40},
    {'name': 'downgrade_tower',   'flat_start': 51,  'flat_end': 61,   'max_targets': 10},
    {'name': 'lightning_storm',   'flat_start': 61,  'flat_end': 66,   'max_targets': 5},
    {'name': 'emp_blaster',       'flat_start': 66,  'flat_end': 71,   'max_targets': 5},
    {'name': 'deflector',         'flat_start': 71,  'flat_end': 76,   'max_targets': 5},
    {'name': 'evasion',           'flat_start': 76,  'flat_end': 81,   'max_targets': 5},
    {'name': 'tech_upgrade',      'flat_start': 81,  'flat_end': 83,   'max_targets': 2},
]

MAX_TOWER_COUNT = 10
MAX_ROUND = 512
ANT_AGE_LIMIT = 64
ANT_MAX_HP = (20, 25, 25)
ANT_GENERATION_CYCLE = (4.0, 4.0, 3.5)
PHEROMONE_SCALE = 10000


REWARD_CONFIG = {
    "base_hp_attack_weight": 2,
    "tower_hp_attack_weight": 0.2,
    "own_coin_gain_weight": 0.05,
    "enemy_coin_gain_weight": 0.01,
    "tower_survival_per_tower": 0.02,
    "enemy_tower_survival_per_tower": 0.01,
    "tower_survival_level_multipliers": [1.0, 1.5, 2.5],
    "balance_reward_weight": 0.02,
    "own_die_penalty_per_ant": -0.05,

    "build_tower_tiers": [0.6, 0.5, 0.5, 0.3, 0.3, 0.3, 0.1],
    "upgrade_tower_l2": 3.0,
    "upgrade_tower_l3": 5.0,
    "downgrade_tower_penalty": -9,
    "upgrade_gen_speed": [15.0, 10.0],
    "upgrade_gen_ant": [7.5, 2.5],
    "deploy_ls": 7.5,
    "deploy_emp": 9.0,
    "deploy_deflector": 1.0,
    "deploy_evasion": 0.8,

    "tech_speed_per_level": 1.5,
    "tech_hp_per_level": 1.0,

    "noop_tolerance": 0,
    "noop_base_penalty": -0.0025,
    "noop_max_penalty": -0.0025,

    "win_reward": 500.0,
    "loss_reward": -500.0,

    "step_reward_clip": 500.0,
}

OBS_NORMALIZATION = {
    "hp_scale": 50.0,
    "coin_scale": 1000.0,
    "distance_scale": 19.0,
    "progress_scale": 64.0,
    "tower_count_scale": 20.0,
    "tower_level_scale": 40.0,
    "tower_spread_scale": 10.0,
    "kill_scale": 20.0,
}
