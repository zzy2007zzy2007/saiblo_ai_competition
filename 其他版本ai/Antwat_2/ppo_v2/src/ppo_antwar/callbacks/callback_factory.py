from typing import Optional

from loguru import logger

from .base_callback import BaseCallback
from .callback_list import CallbackList
from .checkpoint_callback import CheckpointCallback
from .metrics_callback import MetricsLoggingCallback
from .tensorboard_callback import TensorBoardCallback
from ..config.path_config import PathConfig


def create_ppo_callbacks(
    save_interval: int,
    log_interval: int = 10,
    checkpoint_dir: Optional[str] = None,
    path_config: Optional[PathConfig] = None,
    enable_checkpoint: bool = True,
    enable_metrics_logging: bool = True,
    enable_tensorboard: bool = False,
    tensorboard_dir: Optional[str] = None,
    tensorboard_interval: int = 10,
) -> Optional[CallbackList]:
    """回调工厂函数 - 创建标准化的 PPO 回调列表"""
    callbacks = CallbackList()

    if enable_metrics_logging:
        callbacks.append(MetricsLoggingCallback(log_interval=log_interval))

    if enable_checkpoint:
        callbacks.append(
            CheckpointCallback(
                save_interval=save_interval,
                checkpoint_dir=checkpoint_dir,
                path_config=path_config,
            )
        )

    if enable_tensorboard and tensorboard_dir:
        callbacks.append(
            TensorBoardCallback(
                log_dir=tensorboard_dir, log_interval=tensorboard_interval
            )
        )
        logger.info(
            f"TensorBoard logging enabled, output: {tensorboard_dir}, interval: {tensorboard_interval}"
        )

    if len(callbacks) == 0:
        return None
    return callbacks
