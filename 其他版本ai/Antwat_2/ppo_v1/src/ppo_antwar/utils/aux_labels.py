from __future__ import annotations

import numpy as np

HORIZONS = [1, 2, 4, 8, 16]


def compute_aux_labels_from_trajectory(our_snapshots, enemy_snapshots):
    """
    从真实轨迹回看计算辅助标签（我方 + 敌方）。

    Args:
        our_snapshots: List[Tuple[int, int]] — 每步的 (我方塔总HP, 我方累计收入)
        enemy_snapshots: List[Tuple[int, int]] — 每步的 (敌方塔总HP, 敌方累计收入)

    Returns:
        our_tower_labels: List[np.ndarray] — 每个是 shape (5,) 的 float32
        our_gold_labels: List[np.ndarray]  — 每个是 shape (5,) 的 float32
        enemy_tower_labels: List[np.ndarray] — 每个是 shape (5,) 的 float32
        enemy_gold_labels: List[np.ndarray]  — 每个是 shape (5,) 的 float32
    """
    n = len(our_snapshots)
    our_tower_labels = []
    our_gold_labels = []
    enemy_tower_labels = []
    enemy_gold_labels = []

    for t in range(n):
        our_tower_dmg = np.zeros(len(HORIZONS), dtype=np.float32)
        our_gold_inc = np.zeros(len(HORIZONS), dtype=np.float32)
        enemy_tower_dmg = np.zeros(len(HORIZONS), dtype=np.float32)
        enemy_gold_inc = np.zeros(len(HORIZONS), dtype=np.float32)

        cur_our_hp, cur_our_earned = our_snapshots[t]
        cur_enemy_hp, cur_enemy_earned = enemy_snapshots[t]

        for i, h in enumerate(HORIZONS):
            future_idx = min(t + h, n - 1)

            future_our_hp, future_our_earned = our_snapshots[future_idx]
            our_tower_dmg[i] = max(0.0, float(cur_our_hp - future_our_hp))
            our_gold_inc[i] = float(future_our_earned - cur_our_earned)

            future_enemy_hp, future_enemy_earned = enemy_snapshots[future_idx]
            enemy_tower_dmg[i] = max(0.0, float(cur_enemy_hp - future_enemy_hp))
            enemy_gold_inc[i] = float(future_enemy_earned - cur_enemy_earned)

        our_tower_labels.append(our_tower_dmg)
        our_gold_labels.append(our_gold_inc)
        enemy_tower_labels.append(enemy_tower_dmg)
        enemy_gold_labels.append(enemy_gold_inc)

    return our_tower_labels, our_gold_labels, enemy_tower_labels, enemy_gold_labels
