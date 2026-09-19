from typing import Optional

from loguru import logger

from .opponent_selector import OpponentSelector
from .battle_shared_payoff import BattleSharedPayoff
from .opponent_pool import OpponentPool


# ─── 自对弈管理器 ──────────────────────────────────────────────────────────


class SelfPlayManager:
    """自对弈核心管理器 - 整合对手选择、评分更新、可利用性评估。

    对应 OpenRL 的 SelfplayCallback + SelfplayAPI 组合，
    但扩展了 TrueSkill 评分和自适应对手选择。
    """

    def __init__(
        self,
        opponent_pool_size: int,
        min_opponent_games: int,
        exploit_prob: float,
        total_episodes: int = 2000,
    ):
        self._payoff = BattleSharedPayoff(
            decay=0.99, min_win_rate_games=min_opponent_games
        )
        self._pool = OpponentPool(
            max_size=opponent_pool_size,
            min_games_threshold=min_opponent_games,
            payoff=self._payoff,
        )
        self._selector = OpponentSelector(
            payoff=self._payoff,
            opponent_pool=self._pool,
            exploit_prob=exploit_prob,
        )
        self._current_player_id: str = "current"
        self._total_episodes = total_episodes

    def select_opponent(self) -> Optional[str]:
        """从对手池中选择下一个对手"""
        return self._selector.select()

    def add_opponent(self, checkpoint_path: str, episode: int) -> None:
        """添加新对手到对手池和评分矩阵"""
        opponent_id = f"opponent_ep{episode}"
        self._pool.add(opponent_id, checkpoint_path, episode)
        self._payoff.ensure_player(self._current_player_id)
        self._payoff.ensure_player(opponent_id)
        logger.info(f"Added opponent {opponent_id} at episode {episode}")

    def update_payoff(self, opponent_id: str, result: int) -> None:
        """更新对战评分"""
        self._payoff.update(self._current_player_id, opponent_id, result)
        self._pool.increment_games(opponent_id)

    def update_progress(self, episode: int) -> None:
        """更新训练进度"""
        self._selector.update_progress(episode, self._total_episodes)

    def get_opponent_checkpoint_path(self, opponent_id: str) -> Optional[str]:
        """获取指定对手的 checkpoint 路径"""
        return self._pool.get_checkpoint_path(opponent_id)

    def get_opponent_count(self) -> int:
        """获取对手池大小"""
        return len(self._pool.get_all())

    def get_exploitability(self) -> float:
        """计算当前智能体的可利用性评分"""
        return self._payoff.get_exploitability(self._current_player_id)

    def get_statistics(self) -> dict:
        """获取自对弈统计信息"""
        return {
            "pool_size": len(self._pool.get_all()),
            "exploitability": self.get_exploitability(),
            "exploit_prob": self._selector._compute_exploit_prob(),
        }

    def save_state(self, pool_path: str, payoff_path: str) -> None:
        """保存联赛状态"""
        self._pool.save_state(pool_path)
        self._payoff.save_state(payoff_path)

    def load_state(self, pool_path: str, payoff_path: str) -> None:
        """加载联赛状态"""
        self._payoff.load_state(payoff_path)
        self._pool.load_state(pool_path)
        self._pool._payoff = self._payoff
