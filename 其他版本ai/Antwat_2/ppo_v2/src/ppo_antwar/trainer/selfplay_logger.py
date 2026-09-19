from pathlib import Path

from loguru import logger

from ..monitor.constants import SP_LOG_FORMAT


class SelfPlayLogger:
    """自对弈对战日志记录器 - 带有独立自对弈日志文件。

    职责：
    - 向 selfplay/ 目录写入独立 sp_all.log 文件
    - 记录对战生命周期事件
    - 记录对手添加/淘汰事件

    所有日志消息均通过 `logger.bind(category="selfplay")` 标记，
    handler 通过 filter 仅捕获 category="selfplay" 的消息。
    """

    def __init__(self, log_dir: str):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        fmt = SP_LOG_FORMAT
        self._handler = logger.add(
            str(self._log_dir / "sp_all.log"),
            level="DEBUG",
            format=fmt,
            filter=lambda record: record["extra"].get("category") == "selfplay",
        )

    def close(self) -> None:
        try:
            logger.remove(self._handler)
        except ValueError:
            pass

    def log_battle_start(self, episode: int, opponent_id: str, swap: bool) -> None:
        logger.bind(category="selfplay").info(
            f"START episode={episode} vs={opponent_id} swap={swap}"
        )

    def log_battle_end(
        self, episode: int, opponent_id: str, result: str, reward: float, rounds: int
    ) -> None:
        logger.bind(category="selfplay").info(
            f"END episode={episode} vs={opponent_id} result={result} reward={reward:.2f} rounds={rounds}"
        )

    def log_opponent_added(
        self, episode: int, opponent_id: str, pool_size: int
    ) -> None:
        logger.bind(category="selfplay").info(
            f"OPPONENT_ADDED episode={episode} id={opponent_id} pool_size={pool_size}"
        )

    def log_opponent_evicted(
        self, victim_id: str, victim_mu: float, reason: str
    ) -> None:
        logger.bind(category="selfplay").info(
            f"OPPONENT_EVICTED id={victim_id} mu={victim_mu:.2f} reason={reason}"
        )
