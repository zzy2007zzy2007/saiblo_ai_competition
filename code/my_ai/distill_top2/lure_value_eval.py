"""亚军 (saiblo-30th-AI/Game1 cpp_lure_v4) 的 evaluate_terminal —— Python 复刻。

逐条对照来源（不改语义，改动会破坏可蒸馏性）：
  价值公式   其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/include/antgame_ai/
             lure_strategy_v4_terminal_eval.hpp:255-288
  权重常量   .../lure_strategy_v4_params.hpp
  threat     .../lure_strategy_v4_threat.hpp:54-128
  回收价值   .../lure_strategy_v4_core_values.hpp:93-135
             + antgame_cpp_sdk/include/antgame_sdk/random_search_baseline.hpp:833-920
  槽位/核心位 .../antgame_cpp_sdk/include/antgame_sdk/position_slots.hpp:8-169

⚠️ 状态一律是【C++ 引擎】的 GameStateFacade（权威；官方 Python 引擎与 C++ 对不上，
   本文件不依赖它）。全部是纯函数 state -> float，不依赖任何 runner-up C++ 头/编译器。
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_ANT = _REPO / "Ant-Game"
if str(_ANT) not in sys.path:
    sys.path.insert(0, str(_ANT))

from SDK.utils.constants import (  # noqa: E402
    ANT_AGE_LIMIT,
    PLAYER_BASES,
    STRATEGIC_BUILD_ORDER,
    TOWER_BUILD_BASE_COST,
    TOWER_DOWNGRADE_REFUND_RATIO,
    TOWER_STATS,
    AntBehavior,
    AntKind,
    TowerType,
)
from SDK.utils.geometry import hex_distance  # noqa: E402

# ---- params.hpp 里的手写常量（lure_strategy_v4_params.hpp）----
BASE_HP_WEIGHT = 200.0
MONEY_WEIGHT = 10.0
MONEY_DECAY_THRESHOLD = 400.0
MONEY_WEIGHT_ABOVE_THRESHOLD = 8.0
WORKER_THREAT_UNIT = 160.0
COMBAT_BASE_THREAT_UNIT = 300.0
COMBAT_ANCHOR_THREAT_COIN_RATIO = 0.2
COMBAT_ANCHOR_RING_DISTANCE = 1
COMBAT_ANCHOR_RING1_BONUS_RATIO = 0.2
RANDOMIZED_THREAT_SCALE = 0.6
BEWITCHED_THREAT_SCALE = 0.3

LEVEL2_UPGRADE_COST = 60
LEVEL3_UPGRADE_COST = 200


# ── 基础工具 ────────────────────────────────────────────────────────────────
def _max_hp(tower_type) -> int:
    return int(TOWER_STATS[TowerType(int(tower_type))].max_hp)


def _build_cost_for_count(n: int) -> int:
    # constants.py:tower_build_cost_for_count（15 * 3^(n//2)，奇数再 ×2）
    n = max(int(n), 0)
    cost = TOWER_BUILD_BASE_COST * (3 ** (n // 2))
    if n % 2 == 1:
        cost *= 2
    return cost


def _upgrade_cost(tower_type) -> int:
    return LEVEL2_UPGRADE_COST if int(tower_type) < 10 else LEVEL3_UPGRADE_COST


def _destroy_income(tower_count: int, hp: int, max_hp: int) -> int:
    # random_search_baseline.hpp:833 exact_destroy_tower_income
    refund = _build_cost_for_count(max(tower_count - 1, 0)) * TOWER_DOWNGRADE_REFUND_RATIO
    return int(refund * max(hp, 0) / max(1, max_hp))


def _downgrade_income(tower_type, hp: int, max_hp: int) -> int:
    # random_search_baseline.hpp:839 exact_downgrade_tower_income
    refund = _upgrade_cost(tower_type) * TOWER_DOWNGRADE_REFUND_RATIO
    return int(refund * max(hp, 0) / max(1, max_hp))


def _downgrade_target_type(tower_type) -> int:
    t = int(tower_type)
    return 0 if t == 0 else t // 10


def _alive(obj) -> bool:
    hp = getattr(obj, "hp", 0)
    if hp is None or hp <= 0:
        return False
    is_alive = getattr(obj, "is_alive", None)
    if callable(is_alive):
        return bool(is_alive())
    return True


def _too_old(ant) -> bool:
    return int(getattr(ant, "age", 0)) >= ANT_AGE_LIMIT


def _behavior_scale(behavior) -> float:
    b = int(behavior)
    if b == int(AntBehavior.RANDOM):
        return RANDOMIZED_THREAT_SCALE
    if b == int(AntBehavior.BEWITCHED):
        return BEWITCHED_THREAT_SCALE
    return 1.0


def _is_core_build_position(player: int, x: int, y: int) -> bool:
    # position_slots.hpp:154 → code ∈ {C1,C2,C3,L1,L2,L3,R1,R2,R3}
    # code = 槽位在 old_ai_positions 里的 index；index 1..9 正好是这 9 个核心位。
    slots = STRATEGIC_BUILD_ORDER[player]
    try:
        idx = slots.index((x, y))
    except ValueError:
        return False
    return 1 <= idx <= 9


# ── 回收价值 ────────────────────────────────────────────────────────────────
def tower_full_salvage_value(state, player: int) -> float:
    """core_values.hpp:93-131：全部塔降到 basic 再按 HP 降序卖掉。"""
    total = 0.0
    basics: list[int] = []
    for tower in state.towers:
        if tower.player != player or not _alive(tower):
            continue
        cur_type = int(tower.tower_type)
        cur_hp = int(tower.hp)
        while cur_type != 0:
            total += _downgrade_income(cur_type, cur_hp, _max_hp(cur_type))
            prev_max = _max_hp(cur_type)
            cur_type = _downgrade_target_type(cur_type)
            new_max = _max_hp(cur_type)
            cur_hp = (
                max(1, (new_max * cur_hp + prev_max - 1) // prev_max)
                if prev_max > 0
                else new_max
            )
        basics.append(max(cur_hp, 0))
    basics.sort(reverse=True)
    count = len(basics)
    for hp in basics:
        total += _destroy_income(count, hp, _max_hp(0))
        count -= 1
    return total


def tower_estimated_salvage_value(tower, tower_count: int) -> float:
    """random_search_baseline.hpp:898（单塔版，anchor threat 用）。"""
    if not _alive(tower):
        return 0.0
    total = 0.0
    cur_type = int(tower.tower_type)
    cur_hp = int(tower.hp)
    while cur_type != 0:
        total += _downgrade_income(cur_type, cur_hp, _max_hp(cur_type))
        prev_max = _max_hp(cur_type)
        cur_type = _downgrade_target_type(cur_type)
        new_max = _max_hp(cur_type)
        cur_hp = (
            max(1, (new_max * cur_hp + prev_max - 1) // prev_max)
            if prev_max > 0
            else new_max
        )
    total += _destroy_income(tower_count, cur_hp, _max_hp(0))
    return total


# ── 价值分项 ────────────────────────────────────────────────────────────────
def own_equivalent_money_score(equivalent_money: float) -> float:
    # terminal_eval.hpp:83-89
    below = min(equivalent_money, MONEY_DECAY_THRESHOLD)
    above = max(0.0, equivalent_money - MONEY_DECAY_THRESHOLD)
    return below * MONEY_WEIGHT + above * MONEY_WEIGHT_ABOVE_THRESHOLD


def worker_threat_score(state, player: int) -> float:
    # threat.hpp:54-66（rs 版：跳过 too_old / 基地格）
    bx, by = PLAYER_BASES[player]
    threat = 0.0
    for ant in state.ants:
        if ant.player == player or int(ant.kind) != int(AntKind.WORKER):
            continue
        if not _alive(ant) or _too_old(ant) or (ant.x == bx and ant.y == by):
            continue
        dist = max(1, hex_distance(ant.x, ant.y, bx, by))
        threat += WORKER_THREAT_UNIT * max(int(ant.hp), 0) / max(1, int(ant.max_hp)) / dist
    return threat


def combat_anchor_threat_at(state, player: int, x: int, y: int) -> float:
    # threat.hpp:68-88
    threat = 0.0
    tower_count = state.tower_count(player)
    for tower in state.towers:
        if tower.player != player or not _alive(tower):
            continue
        if not _is_core_build_position(player, tower.x, tower.y):
            continue
        dist = max(1, hex_distance(x, y, tower.x, tower.y))
        value = tower_estimated_salvage_value(tower, tower_count)
        tower_threat = value * COMBAT_ANCHOR_THREAT_COIN_RATIO / dist
        if dist <= COMBAT_ANCHOR_RING_DISTANCE:
            tower_threat += value * COMBAT_ANCHOR_RING1_BONUS_RATIO
        threat = max(threat, tower_threat)
    return threat


def combat_threat_at(state, player: int, ant, x: int, y: int) -> float:
    # threat.hpp:90-97
    bx, by = PLAYER_BASES[player]
    dist = max(1, hex_distance(x, y, bx, by))
    threat = COMBAT_BASE_THREAT_UNIT / dist
    threat = max(threat, combat_anchor_threat_at(state, player, x, y))
    threat *= _behavior_scale(ant.behavior)
    return threat


def combat_threat_score(state, player: int) -> float:
    # threat.hpp:117-128
    bx, by = PLAYER_BASES[player]
    threat = 0.0
    for ant in state.ants:
        if ant.player == player or int(ant.kind) != int(AntKind.COMBAT):
            continue
        if not _alive(ant) or _too_old(ant) or (ant.x == bx and ant.y == by):
            continue
        threat += combat_threat_at(state, player, ant, ant.x, ant.y)
    return threat


def evaluate_terminal(state, player: int) -> float:
    """terminal_eval.hpp:255-288（apply_future_threat=False，默认）。"""
    base_hp = int(state.bases[player].hp)
    base_hp_score = float(base_hp) * BASE_HP_WEIGHT
    salvage = tower_full_salvage_value(state, player)
    equivalent_money = float(state.coins[player]) + salvage
    money_score = own_equivalent_money_score(equivalent_money)
    worker_threat = worker_threat_score(state, player)
    combat_threat = combat_threat_score(state, player)
    return base_hp_score + money_score - worker_threat - combat_threat


def eval_breakdown(state, player: int) -> dict:
    """诊断用：返回各分量。"""
    base_hp = int(state.bases[player].hp)
    salvage = tower_full_salvage_value(state, player)
    equiv = float(state.coins[player]) + salvage
    return {
        "base_hp": base_hp,
        "base_hp_score": base_hp * BASE_HP_WEIGHT,
        "coins": float(state.coins[player]),
        "salvage": salvage,
        "equivalent_money": equiv,
        "money_score": own_equivalent_money_score(equiv),
        "worker_threat": worker_threat_score(state, player),
        "combat_threat": combat_threat_score(state, player),
        "total": evaluate_terminal(state, player),
    }
