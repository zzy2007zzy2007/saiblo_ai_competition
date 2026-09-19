import math
import random
from typing import List, Optional

from .battle_shared_payoff import BattleSharedPayoff
from .opponent_pool import OpponentPool
from .trueskill_rating import TrueSkillRating

# ─── 调度参数常量 ──────────────────────────────────────────────────────────
EXPLORE_SCHEDULE_MIN = 0.3
EXPLORE_SCHEDULE_RANGE = 0.6
EXP_DECAY_RATE = 3

# ─── 对手选择器 ────────────────────────────────────────────────────────────


class OpponentSelector:
    """对手选择器 - 实现探索/利用平衡的对手选择策略。

    与 OpenRL 的 Sample Strategy 理念对齐：
    - 利用 (Exploit): 选择胜率最低（最有挑战性）的可信对手
    - 探索 (Explore): 选择不确定性高或随机的对手
    """

    MIN_CONFIDENCE_SIGMA = 5.0
    EXPLOIT_TOP_K = 3

    def __init__(
        self,
        payoff: BattleSharedPayoff,
        opponent_pool: OpponentPool,
        exploit_prob: float,
        adaptive: bool = True,
        schedule: str = "linear",
    ):
        self._payoff = payoff
        self._pool = opponent_pool
        self._exploit_prob = exploit_prob
        self._initial_exploit_prob = exploit_prob
        self._adaptive = adaptive
        self._schedule = schedule
        self._progress: float = 0.0
        self._min_confidence_sigma: float = self.MIN_CONFIDENCE_SIGMA

    def select(self) -> Optional[str]:
        """选择下一个对手

        Returns:
            对手 ID，如果对手池为空则返回 None
        """
        candidates = self._pool.get_all()
        if not candidates:
            return None

        if random.random() < self._compute_exploit_prob():
            return self._exploit_select(candidates)
        else:
            return self._explore_select(candidates)

    def _exploit_select(self, candidates: List[str]) -> str:
        """利用策略：筛选可信对手，按 mu 排序取最强"""
        confident = []
        for cid in candidates:
            rating = self._payoff._trueskill_ratings.get(cid, None)
            if rating is not None and rating.is_confident:
                confident.append(cid)

        if not confident:
            best = min(
                candidates,
                key=lambda cid: (
                    self._payoff._trueskill_ratings.get(cid, TrueSkillRating()).sigma
                ),
            )
            return best

        confident_sorted = sorted(
            confident,
            key=lambda cid: self._payoff._trueskill_ratings[cid].mu,
            reverse=True,
        )
        top_k = min(self.EXPLOIT_TOP_K, len(confident_sorted))
        return confident_sorted[random.randint(0, top_k - 1)]

    def _explore_select(self, candidates: List[str]) -> str:
        """探索策略：选择 sigma 较大（不确定性高）的对手"""
        high_sigma = [
            cid
            for cid in candidates
            if self._payoff._trueskill_ratings.get(cid, TrueSkillRating()).sigma
            > self._min_confidence_sigma
        ]

        if not high_sigma:
            high_sigma = candidates

        return random.choice(high_sigma)

    def select_batch(self, num_selections: int) -> List[str]:
        """批量选择多个不重复的对手"""
        selected = []
        available = set(self._pool.get_all())
        for _ in range(num_selections):
            if not available:
                break
            candidates = list(available)
            if random.random() < self._compute_exploit_prob():
                opponent = self._exploit_select(candidates)
            else:
                opponent = self._explore_select(candidates)
            selected.append(opponent)
            available.discard(opponent)
        return selected

    def update_progress(self, episode: int, total_episodes: int) -> None:
        """更新训练进度，动态调整探索/利用概率

        支持 linear / sigmoid / exp 三种增长模式
        """
        self._progress = episode / max(total_episodes, 1)

    def _compute_exploit_prob(self) -> float:
        """根据当前进度和调度策略计算利用概率"""
        if not self._adaptive:
            return self._exploit_prob

        p = self._progress
        if self._schedule == "sigmoid":
            x = (p - 0.5) * 10
            p_effective = 1.0 / (1.0 + math.exp(-x))
        elif self._schedule == "exp":
            p_effective = EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * (
                1.0 - math.exp(-EXP_DECAY_RATE * p)
            )
        else:
            p_effective = EXPLORE_SCHEDULE_MIN + EXPLORE_SCHEDULE_RANGE * p

        return p_effective
