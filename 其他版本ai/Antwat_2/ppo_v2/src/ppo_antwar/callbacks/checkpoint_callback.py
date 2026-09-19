from pathlib import Path
from typing import List, Optional

from loguru import logger

from .base_ppo_callback import BasePPOCallback
from ..config.path_config import PathConfig


class CheckpointCallback(BasePPOCallback):
    """检查点保存回调 - 定期保存模型检查点

    对应 OpenRL 的 CheckpointCallback，扩展了旧检查点清理逻辑。
    """

    def __init__(
        self,
        save_interval: int,
        checkpoint_dir: Optional[str] = None,
        path_config: Optional[PathConfig] = None,
        keep_last_n: int = 5,
    ):
        super().__init__()
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.path_config = path_config
        self.keep_last_n = keep_last_n
        self._step_count = 0
        self._saved_checkpoints: List[str] = []

    def _on_step(self) -> bool:
        """在训练步骤触发时检查是否需要保存检查点"""
        if self.checkpoint_dir is not None and self._trainer_ref is not None:
            episode = self._step_count
            if episode % self.save_interval == 0:
                checkpoint_path = str(
                    Path(self.checkpoint_dir) / f"checkpoint_ep{episode}.pt"
                )
                Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                self._trainer_ref.save_checkpoint(checkpoint_path, episode, {})
                self._saved_checkpoints.append(checkpoint_path)
                logger.info(f"Checkpoint saved at episode {episode}")
                self._cleanup_old_checkpoints()
            self._step_count = episode + 1
        return True

    def _cleanup_old_checkpoints(self):
        """清理过期检查点，只保留最近 N 个"""
        if len(self._saved_checkpoints) > self.keep_last_n:
            for old_path in self._saved_checkpoints[: -self.keep_last_n]:
                try:
                    Path(old_path).unlink()
                    logger.debug(f"Removed old checkpoint: {old_path}")
                except FileNotFoundError:
                    pass
            self._saved_checkpoints = self._saved_checkpoints[-self.keep_last_n :]
