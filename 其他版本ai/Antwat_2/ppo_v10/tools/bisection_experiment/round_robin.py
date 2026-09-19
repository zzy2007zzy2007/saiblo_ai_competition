"""RoundRobin — 全量循环赛（Ground Truth 计算 & 精英精排）

用于:
  1. 计算全量 RR 的 TrueSkill 排名（作为另一种 baseline 对比）
  2. Top16 精英精排（v4 §2 的最后步骤）
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

try:
    from .bisection_coordinator import SimplePayoff
    from .individual import Individual, simulate_battle
except ImportError:
    from bisection_coordinator import SimplePayoff  # type: ignore
    from individual import Individual, simulate_battle  # type: ignore


def run_round_robin(
    individuals: Dict[str, Individual],
    participant_ids: List[str],
    n_battles: int = 4,
    payoff: Optional[SimplePayoff] = None,
    max_workers: int = 25,
    seed: int = 42,
) -> Tuple[SimplePayoff, List[str], int]:
    """对指定参与者执行全对 RoundRobin

    Args:
        individuals: 所有个体 {id: Individual}
        participant_ids: 参与 RR 的个体 ID 列表
        n_battles: 每对对战局数（交替先后手，实际为 n_battles*2）
        payoff: 可选的已有 payoff（延续评分），None 则新建
        max_workers: 并行数
        seed: 随机种子

    Returns:
        (payoff, ranking, total_battles)
    """
    if payoff is None:
        payoff = SimplePayoff()

    for pid in participant_ids:
        payoff.ensure_player(pid)

    rng = np.random.RandomState(seed)
    total_battles = 0
    n = len(participant_ids)
    total_pairs = n * (n - 1) // 2

    logger.info(f"RoundRobin: {n} 人, {total_pairs} 对, "
                 f"每对 {n_battles * 2} 局 = 预计 {total_pairs * n_battles * 2} 局")

    for i in range(n):
        for j in range(i + 1, n):
            id_a = participant_ids[i]
            id_b = participant_ids[j]
            ind_a = individuals[id_a]
            ind_b = individuals[id_b]

            for game_idx in range(n_battles * 2):
                # 交替先后手
                first = id_a if game_idx % 2 == 0 else id_b
                second = id_b if game_idx % 2 == 0 else id_a
                ind_first = individuals[first]
                ind_second = individuals[second]

                outcome = simulate_battle(ind_first, ind_second, rng)
                result_code = outcome.to_result_code(first, second)
                payoff.update_match(first, second, result_code)
                total_battles += 1

    # 排名
    ranking = sorted(
        participant_ids,
        key=lambda pid: payoff.get_trueskill_mu(pid),
        reverse=True,
    )

    return payoff, ranking, total_battles
