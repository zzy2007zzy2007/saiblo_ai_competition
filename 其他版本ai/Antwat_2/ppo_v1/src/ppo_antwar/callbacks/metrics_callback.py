import numpy as np
from .base_ppo_callback import BasePPOCallback


class MetricsLoggingCallback(BasePPOCallback):
    """训练指标记录回调 - 统一训练日志记录"""

    def __init__(
        self,
        log_interval: int = 10,
        window_size: int = 100,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.window_size = window_size

    def _on_step(self) -> bool:
        episode = self.trainer.episode_count
        if episode % self.log_interval == 0:
            episode_rewards = getattr(self.trainer, "episode_rewards", [])
            episode_lengths = getattr(self.trainer, "episode_lengths", [])

            recent_rewards = episode_rewards[-self.window_size:]
            recent_lengths = episode_lengths[-self.window_size:]

            if recent_rewards:
                mean_reward = np.mean(recent_rewards)
                mean_length = np.mean(recent_lengths) if recent_lengths else 0

                if self.verbose >= 1:
                    print(f"Episode {episode}: mean_reward={mean_reward:.3f}, mean_length={mean_length:.1f}")

                if self.trainer.logger:
                    self.trainer.logger.log_training_details(
                        episode=episode,
                        avg_reward=mean_reward,
                        avg_loss=0,
                        avg_length=mean_length,
                        steps=self.trainer.total_steps,
                    )
        return True
