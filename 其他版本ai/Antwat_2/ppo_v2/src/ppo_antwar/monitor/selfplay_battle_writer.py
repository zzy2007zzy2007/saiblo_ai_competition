"""SelfPlay 每局对战数据写入器 - 写入 selfplay_battle_log.jsonl"""

import time
from pathlib import Path
from .episode_record import EpisodeRecord


class SelfPlayBattleWriter:
    """自对弈每局对战数据写入器。

    职责：
    - 每局结束后将结构化数据追加写入 selfplay_battle_log.jsonl
    - 替代原有的 _per_episode_stats_buffer + _flush_per_episode_stats() 模式
    """

    def __init__(self, filepath: str):
        self._filepath = Path(filepath)
        self._filepath.parent.mkdir(parents=True, exist_ok=True)

    def write_episode(self, record: EpisodeRecord) -> None:
        """写入单局对战数据"""
        with open(self._filepath, "a") as f:
            f.write(record.to_json() + "\n")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


__all__ = ["SelfPlayBattleWriter"]
