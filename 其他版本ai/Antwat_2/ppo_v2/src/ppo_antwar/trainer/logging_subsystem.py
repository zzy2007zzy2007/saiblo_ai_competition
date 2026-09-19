from dataclasses import dataclass
from ..monitor.time_tracker import TimeTracker
from ..monitor.logger import Logger
from ..monitor.selfplay_battle_writer import SelfPlayBattleWriter
from ..monitor.batch_metrics_writer import BatchMetricsWriter
from ..monitor.episode_batch_writer import EpisodeBatchWriter
from ..monitor.eval_battle_writer import EvalBattleWriter
from ..monitor.system_metrics_sampler import SystemMetricsSampler
from .selfplay_logger import SelfPlayLogger


@dataclass
class LoggingSubsystem:
    """日志子系统 - 封装所有 logger/writer 实例，提供统一生命周期管理。"""

    loguru_handler_id: int
    time_tracker: TimeTracker
    logger: Logger
    selfplay_logger: SelfPlayLogger
    battle_logger: SelfPlayBattleWriter
    batch_logger: BatchMetricsWriter
    ep_batch_logger: EpisodeBatchWriter
    eval_logger: EvalBattleWriter
    system_sampler: SystemMetricsSampler

    def close_all(self) -> None:
        """统一的关闭入口，按依赖顺序关闭所有组件。"""
        if self.battle_logger:
            self.battle_logger.close()
        if self.batch_logger:
            self.batch_logger.close()
        if self.ep_batch_logger:
            self.ep_batch_logger.close()
        if self.eval_logger:
            self.eval_logger.close()
        self.selfplay_logger.close()
        self.system_sampler.stop()
        try:
            from loguru import logger as loguru_logger

            loguru_logger.remove(self.loguru_handler_id)
        except (ValueError, AttributeError):
            pass


__all__ = ["LoggingSubsystem"]
