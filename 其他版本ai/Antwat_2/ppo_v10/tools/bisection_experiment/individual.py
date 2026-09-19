"""个体模拟器与战局生成

每个个体有两个参数:
  - p: 真实战力 (true skill)，决定 Ground Truth 排名
  - q: 表现方差 (performance std)，p 相近时 q 小的更稳定

两个个体对战时，实际表现从正态分布采样:
  perf_A ~ N(p_A, q_A^2)
  perf_B ~ N(p_B, q_B^2)

A 胜出 iff perf_A > perf_B
等价于 P(A胜) = Φ((p_A - p_B) / sqrt(q_A^2 + q_B^2))

这与 TrueSkill 的底层假设完全一致，确保了测试的公平性。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Individual:
    """模拟个体"""
    id: str
    p: float            # 真实战力 (ground truth)
    q: float            # 表现标准差 (方差越大越不稳定)

    @property
    def true_rank_key(self) -> float:
        """Ground Truth 排名的 key: p 越大越强"""
        return self.p


class BattleOutcome:
    """单局对战结果"""
    def __init__(self, winner_id: Optional[str], player_a: str, player_b: str):
        self.winner_id = winner_id   # None = 平局
        self.player_a = player_a
        self.player_b = player_b

    def to_result_code(self, home: str, away: str) -> int:
        """转为 TrueSkill update 用的 result code: 1=home胜, -1=away胜, 0=平"""
        if self.winner_id is None:
            return 0
        if self.winner_id == home:
            return 1
        return -1


def generate_individuals(
    n: int,
    p_dist: str = "uniform",
    q_mode: str = "constant",
    q_value: float = 8.33,
    p_min: float = 10.0,
    p_max: float = 90.0,
    p_mean: float = 50.0,
    p_std: float = 20.0,
    seed: int = 42,
) -> List[Individual]:
    """生成 n 个模拟个体

    Args:
        n: 个体数量
        p_dist: p 的分布 ("uniform" | "normal" | "bimodal")
        q_mode: q 的模式 ("constant" | "proportional" | "random")
        q_value: q 的基础值 (constant 模式下的固定值)
        p_min, p_max: uniform 分布的范围
        p_mean, p_std: normal 分布的参数
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)

    # 生成 p
    if p_dist == "uniform":
        p_vals = rng.uniform(p_min, p_max, n)
    elif p_dist == "normal":
        p_vals = rng.normal(p_mean, p_std, n)
        p_vals = np.clip(p_vals, 0.0, 100.0)
    elif p_dist == "bimodal":
        half = n // 2
        p_vals = np.concatenate([
            rng.normal(p_mean - p_std, p_std * 0.5, half),
            rng.normal(p_mean + p_std, p_std * 0.5, n - half),
        ])
        p_vals = np.clip(p_vals, 0.0, 100.0)
    else:
        raise ValueError(f"Unknown p_dist: {p_dist}")

    # 生成 q
    if q_mode == "constant":
        q_vals = np.full(n, q_value)
    elif q_mode == "proportional":
        # q 与 p 成正比: 强者更稳定? 还是更不稳定? 默认越强越稳定
        q_vals = q_value * (1.0 - 0.5 * p_vals / 100.0)
        q_vals = np.clip(q_vals, 1.0, q_value * 2)
    elif q_mode == "random":
        q_vals = rng.uniform(2.0, q_value, n)
    else:
        raise ValueError(f"Unknown q_mode: {q_mode}")

    individuals = [
        Individual(id=f"ind_{i:04d}", p=float(p_vals[i]), q=float(q_vals[i]))
        for i in range(n)
    ]
    return individuals


def simulate_battle(
    ind_a: Individual,
    ind_b: Individual,
    rng: Optional[np.random.RandomState] = None,
) -> BattleOutcome:
    """模拟两个个体的一局对战

    性能从 N(p, q^2) 采样，perform_a > perform_b → A 胜

    Args:
        ind_a, ind_b: 对战双方
        rng: 随机数生成器

    Returns:
        BattleOutcome 对象
    """
    if rng is None:
        rng = np.random.RandomState()

    # A 胜概率 = Φ((p_A - p_B) / sqrt(q_A^2 + q_B^2))
    diff = ind_a.p - ind_b.p
    combined_std = math.sqrt(ind_a.q ** 2 + ind_b.q ** 2)

    if combined_std < 1e-9:
        # 确定性: p 高者胜
        if diff > 0:
            winner_id = ind_a.id
        elif diff < 0:
            winner_id = ind_b.id
        else:
            winner_id = None  # 完全平手
    else:
        # P(A胜) = Φ(diff / combined_std)
        prob_a_win = 0.5 * (1.0 + math.erf(diff / (combined_std * math.sqrt(2))))

        # 平局概率 (模拟真实比赛中罕见的平局)
        draw_prob = 0.01
        rand_val = rng.uniform(0, 1)

        if rand_val < draw_prob:
            winner_id = None
        elif rand_val < draw_prob + (1 - draw_prob) * prob_a_win:
            winner_id = ind_a.id
        else:
            winner_id = ind_b.id

    return BattleOutcome(
        winner_id=winner_id,
        player_a=ind_a.id,
        player_b=ind_b.id,
    )


def get_ground_truth_ranking(individuals: List[Individual]) -> List[str]:
    """按 p 降序排列，得到 Ground Truth 排名"""
    sorted_inds = sorted(individuals, key=lambda x: x.p, reverse=True)
    return [ind.id for ind in sorted_inds]
