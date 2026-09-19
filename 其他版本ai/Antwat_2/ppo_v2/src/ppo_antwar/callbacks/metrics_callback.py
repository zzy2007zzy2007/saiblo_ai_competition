from .base_ppo_callback import BasePPOCallback


class MetricsLoggingCallback(BasePPOCallback):
    """训练指标记录回调 - 定期输出训练指标摘要"""

    def __init__(self, log_interval: int = 10, window_size: int = 100):
        super().__init__()
        self.log_interval = log_interval
        self.window_size = window_size
        self._step_count = 0
        self._last_log_step = 0

    def _on_step(self) -> bool:
        """在训练步骤触发时检查是否需要记录指标"""
        self._step_count += 1

        metrics = self.get_last_metrics()
        if metrics and (self._step_count - self._last_log_step) >= self.log_interval:
            from loguru import logger

            training_loss = metrics.get('training_loss') or 0
            policy_loss = metrics.get('policy_loss') or 0
            value_loss = metrics.get('value_loss') or 0
            entropy = metrics.get('entropy') or 0
            reward_mean = metrics.get('reward_mean') or 0

            logger.info(
                f"Episode {self._step_count} | "
                f"loss={training_loss:.4f} "
                f"policy_loss={policy_loss:.4f} "
                f"value_loss={value_loss:.4f} "
                f"entropy={entropy:.4f} "
                f"reward_mean={reward_mean:.2f}"
            )
            self._last_log_step = self._step_count
        return True
