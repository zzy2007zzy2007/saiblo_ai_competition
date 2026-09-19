"""TensorBoard 日志回调"""

from typing import Any, Optional

from .base_ppo_callback import BasePPOCallback
from ..trainer.metrics_schema import MetricsSchema


class TensorBoardCallback(BasePPOCallback):
    """记录训练指标到 TensorBoard，便于浏览器可视化监控"""

    def __init__(self, log_dir: str, log_interval: int = 10):
        super().__init__()
        self._log_dir = log_dir
        self._log_interval = log_interval
        self._step_count = 0
        self._writer: Optional[Any] = None

    @property
    def _tb_writer(self):
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter
            import os

            os.makedirs(self._log_dir, exist_ok=True)
            self._writer = SummaryWriter(self._log_dir)
        return self._writer

    def _on_step(self) -> bool:
        self._step_count += 1
        if (self._step_count - 1) % self._log_interval != 0:
            return True
        if self._trainer_ref is None:
            return True

        metrics = self.get_last_metrics()
        if not metrics:
            return True

        writer = self._tb_writer

        for key in MetricsSchema.get_tensorboard_keys():
            if key in metrics:
                writer.add_scalar(key, metrics[key], self._step_count)

        return True

    def on_training_end(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
