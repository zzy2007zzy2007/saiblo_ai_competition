from typing import Optional, List

from .metrics_callback import MetricsLoggingCallback
from .checkpoint_callback import CheckpointCallback
from ppo_antwar.config.path_config import PathConfig


class CallbackList:
    """简单的回调列表实现"""

    def __init__(self, callbacks: List = None):
        self.callbacks = callbacks or []

    def append(self, callback) -> None:
        self.callbacks.append(callback)

    def init_callback(self, trainer) -> None:
        for callback in self.callbacks:
            if hasattr(callback, 'init_callback'):
                callback.init_callback(trainer)

    def on_training_start(self) -> None:
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start()

    def on_training_end(self) -> None:
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end()

    def on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            if hasattr(callback, 'on_step'):
                continue_training = callback.on_step() and continue_training
        return continue_training


def create_ppo_callbacks(
    save_interval: int = 1000,
    log_interval: int = 10,
    checkpoint_dir: str = None,
    path_config: Optional[PathConfig] = None,
    enable_checkpoint: bool = True,
    enable_metrics_logging: bool = True,
) -> Optional[CallbackList]:
    """PPO回调工厂 - 创建标准化的回调列表
    
    Args:
        save_interval: checkpoint 保存间隔
        path_config: 路径配置对象（优先使用）
    """

    callbacks = []

    if enable_metrics_logging:
        callbacks.append(
            MetricsLoggingCallback(log_interval=log_interval, verbose=1)
        )

    if enable_checkpoint:
        if path_config:
            callbacks.append(
                CheckpointCallback(
                    save_interval=save_interval,
                    path_config=path_config,
                    verbose=1,
                )
            )
        elif checkpoint_dir:
            callbacks.append(
                CheckpointCallback(
                    save_interval=save_interval,
                    checkpoint_dir=checkpoint_dir,
                    verbose=1,
                )
            )
        else:
            default_path_config = PathConfig()
            callbacks.append(
                CheckpointCallback(
                    save_interval=save_interval,
                    path_config=default_path_config,
                    verbose=1,
                )
            )

    return CallbackList(callbacks) if callbacks else None
