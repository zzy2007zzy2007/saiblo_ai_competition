from __future__ import annotations
from typing import List, Dict, Tuple

import numpy as np

from .action_constants import NUM_HORIZONS, AUX_HORIZON_STEPS


def compute_aux_labels_from_trajectory(
    step_aux_data: List[Dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从 per-round 辅助原始值计算多视界累积标签。

    输入 `step_aux_data` 的每个元素是一个 dict:
        own_twr_dmg  : float  # 本回合己方塔受到的伤害（= mid.hp − new.hp）
        enemy_twr_dmg: float  # 本回合敌方塔受到的伤害
        own_gold     : float  # 本回合己方金币毛收入（= new.coins − mid.coins）
        enemy_gold   : float  # 本回合敌方金币毛收入
        own_base_dmg : float  # 本回合己方基地受到的伤害
        enemy_base_dmg: float # 本回合敌方基地受到的伤害

    输出:
        tower_damage_labels: (T, 10) float32  [own_5h, enemy_5h]
        gold_income_labels:  (T, 10) float32  [own_5h, enemy_5h]
        base_damage_labels:  (T, 10) float32  [own_5h, enemy_5h]
        其中 T = len(step_aux_data) − max(AUX_HORIZON_STEPS)
    """
    max_horizon = max(AUX_HORIZON_STEPS)
    n = len(step_aux_data)
    T = n - max_horizon
    if T <= 0:
        return (
            np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
            np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
            np.zeros((0, NUM_HORIZONS * 2), dtype=np.float32),
        )

    # 前缀和：O(n) 预处理 + O(T × |H|) 查询
    pref_own_twr = np.zeros(n + 1, dtype=np.float32)
    pref_enemy_twr = np.zeros(n + 1, dtype=np.float32)
    pref_own_gold = np.zeros(n + 1, dtype=np.float32)
    pref_enemy_gold = np.zeros(n + 1, dtype=np.float32)
    pref_own_base = np.zeros(n + 1, dtype=np.float32)
    pref_enemy_base = np.zeros(n + 1, dtype=np.float32)

    for i, d in enumerate(step_aux_data):
        pref_own_twr[i + 1] = pref_own_twr[i] + d["own_twr_dmg"]
        pref_enemy_twr[i + 1] = pref_enemy_twr[i] + d["enemy_twr_dmg"]
        pref_own_gold[i + 1] = pref_own_gold[i] + d["own_gold"]
        pref_enemy_gold[i + 1] = pref_enemy_gold[i] + d["enemy_gold"]
        pref_own_base[i + 1] = pref_own_base[i] + d.get("own_base_dmg", 0.0)
        pref_enemy_base[i + 1] = pref_enemy_base[i] + d.get("enemy_base_dmg", 0.0)

    tower_labels = np.zeros((T, NUM_HORIZONS * 2), dtype=np.float32)
    gold_labels = np.zeros((T, NUM_HORIZONS * 2), dtype=np.float32)
    base_labels = np.zeros((T, NUM_HORIZONS * 2), dtype=np.float32)
    H = len(AUX_HORIZON_STEPS)

    for t in range(T):
        for h_idx, h in enumerate(AUX_HORIZON_STEPS):
            # own 部分 (前 5 维)
            tower_labels[t, h_idx] = pref_own_twr[t + h] - pref_own_twr[t]
            gold_labels[t, h_idx] = pref_own_gold[t + h] - pref_own_gold[t]
            base_labels[t, h_idx] = pref_own_base[t + h] - pref_own_base[t]
            # enemy 部分 (后 5 维)
            e = h_idx + H
            tower_labels[t, e] = pref_enemy_twr[t + h] - pref_enemy_twr[t]
            gold_labels[t, e] = pref_enemy_gold[t + h] - pref_enemy_gold[t]
            base_labels[t, e] = pref_enemy_base[t + h] - pref_enemy_base[t]

    return tower_labels, gold_labels, base_labels
