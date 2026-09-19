import json
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EpisodeRecord:
    """自对弈单局对战数据结构"""

    episode: int
    swap: bool
    opponent_id: str
    result: str
    rounds: int
    reward: float
    length: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    end_own_hp: float = 0.0
    end_own_coins: float = 0.0
    max_own_coins: float = 0.0
    total_own_coin_income: float = 0.0
    end_enemy_hp: float = 0.0
    end_enemy_coins: float = 0.0
    max_enemy_coins: float = 0.0
    total_enemy_coin_income: float = 0.0
    action_counts: Optional[Dict[str, int]] = None
    action_rewards: Optional[Dict[str, float]] = None
    reward_sources: Optional[Dict[str, float]] = None

    def to_json(self) -> str:
        """将记录序列化为 JSON 行"""
        now = time.time()
        record = {
            "episode": self.episode,
            "swap": self.swap,
            "opponent_id": self.opponent_id,
            "result": self.result,
            "rounds": self.rounds,
            "reward": round(self.reward, 4),
            "length": self.length,
            "start_time": self.start_time if self.start_time is not None else now,
            "end_time": self.end_time if self.end_time is not None else now,
            "duration": round(self.duration if self.duration is not None else 0.0, 3),
            "end_own_hp": round(self.end_own_hp, 2),
            "end_own_coins": round(self.end_own_coins, 2),
            "max_own_coins": round(self.max_own_coins, 2),
            "total_own_coin_income": round(self.total_own_coin_income, 2),
            "end_enemy_hp": round(self.end_enemy_hp, 2),
            "end_enemy_coins": round(self.end_enemy_coins, 2),
            "max_enemy_coins": round(self.max_enemy_coins, 2),
            "total_enemy_coin_income": round(self.total_enemy_coin_income, 2),
            "action_counts": self.action_counts or {},
            "action_rewards": {
                k: round(v, 4) for k, v in (self.action_rewards or {}).items()
            },
            "reward_sources": {
                k: round(v, 4) for k, v in (self.reward_sources or {}).items()
            },
        }
        return json.dumps(record, ensure_ascii=False)


__all__ = ["EpisodeRecord"]
