from typing import Tuple

import trueskill


# ─── TrueSkill 评分 ────────────────────────────────────────────────────────


class TrueSkillRating:
    """封装单个玩家的 TrueSkill 评分"""

    _CONFIDENT_SIGMA = 3.0

    def __init__(self, mu: float = 25.0, sigma: float = 8.33, games_played: int = 0):
        self.mu = mu
        self.sigma = sigma
        self.games_played = games_played

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return (self.mu - 3 * self.sigma, self.mu + 3 * self.sigma)

    @property
    def is_confident(self) -> bool:
        return self.sigma < self._CONFIDENT_SIGMA

    def to_trueskill(self) -> trueskill.Rating:
        return trueskill.Rating(mu=self.mu, sigma=self.sigma)

    @classmethod
    def from_trueskill(cls, rating: trueskill.Rating, games_played: int = 0) -> "TrueSkillRating":
        obj = cls(mu=rating.mu, sigma=rating.sigma)
        obj.games_played = games_played
        return obj
