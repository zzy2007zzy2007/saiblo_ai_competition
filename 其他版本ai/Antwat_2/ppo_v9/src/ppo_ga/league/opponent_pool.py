import json
from typing import Dict, List, Optional

from loguru import logger

from .trueskill_rating import TrueSkillRating


# ─── 对手池 ────────────────────────────────────────────────────────────────


class OpponentPool:
    """对手池管理器 - 管理对手的添加、淘汰和状态持久化"""

    _EVICTION_PROTECTION_FACTOR = 0.5

    def __init__(
        self,
        max_size: int,
        min_games_threshold: int,
        payoff: Optional["BattleSharedPayoff"] = None,
    ):
        self.max_size = max_size
        self.min_games_threshold = min_games_threshold
        self._opponents: List[str] = []
        self._checkpoint_paths: Dict[str, str] = {}
        self._games_played: Dict[str, int] = {}
        self._added_episodes: Dict[str, int] = {}
        self._payoff = payoff

    def add(self, opponent_id: str, checkpoint_path: str, episode: int) -> None:
        """添加新对手 - 池满时触发淘汰"""
        if opponent_id in self._opponents:
            return

        self._checkpoint_paths[opponent_id] = checkpoint_path
        self._added_episodes[opponent_id] = episode
        self._games_played.setdefault(opponent_id, 0)

        if len(self._opponents) >= self.max_size:
            self._evict_worst()

        self._opponents.append(opponent_id)
        logger.info(
            f"Opponent {opponent_id} added to pool (size: {len(self._opponents)})"
        )

    def _evict_worst(self) -> str:
        """淘汰最差对手

        淘汰评分 = mu + protection_bonus
        protection_bonus = max(0, min_games_threshold - games) * 0.5
        """
        if not self._opponents:
            return ""

        evictable = []
        for opp_id in self._opponents:
            games = self._games_played.get(opp_id, 0)
            if games < self.min_games_threshold:
                continue
            evictable.append(opp_id)

        if not evictable:
            evictable = self._opponents

        def eviction_score(opp_id):
            mu = 25.0
            if self._payoff is not None:
                rating = self._payoff._trueskill_ratings.get(opp_id, TrueSkillRating())
                mu = rating.mu
            games = self._games_played.get(opp_id, 0)
            protection = (
                max(0, self.min_games_threshold - games)
                * self._EVICTION_PROTECTION_FACTOR
            )
            return mu + protection

        victim = min(evictable, key=eviction_score)
        self._opponents.remove(victim)
        self._checkpoint_paths.pop(victim, None)
        self._games_played.pop(victim, None)
        self._added_episodes.pop(victim, None)
        logger.info(f"Opponent {victim} evicted from pool")
        return victim

    def increment_games(self, opponent_id: str) -> None:
        """增加对战场次计数"""
        self._games_played[opponent_id] = self._games_played.get(opponent_id, 0) + 1

    def get_checkpoint_path(self, opponent_id: str) -> str:
        """获取对手的检查点路径"""
        return self._checkpoint_paths.get(opponent_id, "")

    def get_all(self) -> List[str]:
        """获取所有对手 ID 列表"""
        return list(self._opponents)

    def save_state(self, filepath: str) -> None:
        """保存对手池状态到 JSON"""
        state = {
            "opponents": self._opponents,
            "checkpoint_paths": self._checkpoint_paths,
            "games_played": self._games_played,
            "added_episodes": self._added_episodes,
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)
        logger.info(f"Pool state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """从 JSON 加载对手池状态"""
        with open(filepath, "r") as f:
            state = json.load(f)
        required_fields = ["opponents", "checkpoint_paths", "games_played", "added_episodes"]
        for field in required_fields:
            if field not in state:
                raise ValueError(f"Pool state missing required field: {field}")
        self._opponents = state["opponents"]
        self._checkpoint_paths = state["checkpoint_paths"]
        self._games_played = state["games_played"]
        self._added_episodes = state["added_episodes"]
        logger.info(f"Pool state loaded from {filepath}")
