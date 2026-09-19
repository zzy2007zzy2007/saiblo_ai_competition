import json
import time
from typing import Any, Dict, Optional


class TimeTracker:
    """时间追踪器 - 追踪训练和对战过程中的时间消耗"""

    def __init__(self, max_episode_times: int = 1000):
        self._start_time: Optional[float] = None
        self._episode_times: list = []
        self._max_episode_times = max_episode_times
        self._episode_count: int = 0

        self._training_start: Optional[float] = None
        self._training_end: Optional[float] = None
        self._battle_count: int = 0
        self._battle_time_total: float = 0.0
        self._last_battle_start: Optional[float] = None
        self._last_battle_end: Optional[float] = None
        self._last_battle_duration: float = 0.0

    def start(self) -> None:
        self._start_time = time.time()

    def record_episode(self, duration: float) -> None:
        self._episode_times.append(duration)
        if len(self._episode_times) > self._max_episode_times:
            self._episode_times = self._episode_times[-self._max_episode_times :]
        self._episode_count += 1

    def start_training(self) -> None:
        self._training_start = time.time()

    def end_training(self) -> None:
        self._training_end = time.time()

    def start_battle(self) -> None:
        self._last_battle_start = time.time()

    def end_battle(self) -> None:
        self._last_battle_end = time.time()
        if self._last_battle_start is not None:
            duration = self._last_battle_end - self._last_battle_start
            self._last_battle_duration = duration
            self._battle_time_total += duration
            self._battle_count += 1

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def episodes_per_minute(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return (self._episode_count / self.elapsed_seconds) * 60.0

    @property
    def avg_episode_time(self) -> float:
        if not self._episode_times:
            return 0.0
        return sum(self._episode_times) / len(self._episode_times)

    def get_summary(self) -> dict:
        return {
            "elapsed_seconds": self.elapsed_seconds,
            "episodes_per_minute": self.episodes_per_minute,
            "avg_episode_time": self.avg_episode_time,
            "total_episodes": self._episode_count,
        }

    @staticmethod
    def _format_time(t: Optional[float]) -> Optional[str]:
        if t is None:
            return None
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))

    def save_time_statistics(self, path: str) -> Dict[str, Any]:
        stats = {
            "training_episodes_count": self._episode_count,
            "training_time_total": self.elapsed_seconds,
            "battle_evaluations_count": self._battle_count,
            "battle_time_total": self._battle_time_total,
            "last_training_start_time": self._format_time(self._training_start),
            "last_training_end_time": self._format_time(self._training_end),
            "last_battle_start_time": self._format_time(self._last_battle_start),
            "last_battle_end_time": self._format_time(self._last_battle_end),
            "last_battle_duration": self._last_battle_duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        return stats
