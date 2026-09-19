import os
from typing import Optional
from .base_ppo_callback import BasePPOCallback

from ppo_antwar.config.path_config import PathConfig


class CheckpointCallback(BasePPOCallback):
    """检查点保存回调 - 替代手动 episode % save_interval 判断"""

    def __init__(
        self,
        save_interval: int = 1000,
        path_config: PathConfig = None,
        checkpoint_dir: str = None,
        keep_last_n: int = 5,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self._saved_checkpoints = []
        
        if path_config is not None:
            self.checkpoint_dir = str(path_config.checkpoint_dir)
        elif checkpoint_dir is not None:
            self.checkpoint_dir = checkpoint_dir
        else:
            raise ValueError("Either path_config or checkpoint_dir is required")

    def _on_step(self) -> bool:
        episode = self.trainer.episode_count
        if episode > 0 and episode % self.save_interval == 0:
            checkpoint_path = self.trainer.save_checkpoint(
                f"checkpoint_{episode}.pt"
            )
            self._saved_checkpoints.append(checkpoint_path)
            self._cleanup_old_checkpoints()

            if self.verbose >= 1:
                print(f"Saved checkpoint: {checkpoint_path}")

            if self.trainer.logger:
                self.trainer.logger.log_model_save(checkpoint_path)
        return True

    def _cleanup_old_checkpoints(self) -> None:
        """保留最近N个检查点，删除旧的"""
        if len(self._saved_checkpoints) > self.keep_last_n:
            old_path = self._saved_checkpoints.pop(0)
            if old_path and old_path.endswith(".pt"):
                try:
                    os.remove(old_path)
                except OSError:
                    pass