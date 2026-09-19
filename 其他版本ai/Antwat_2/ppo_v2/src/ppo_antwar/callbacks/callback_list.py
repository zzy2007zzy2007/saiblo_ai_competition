from typing import List
from .base_callback import BaseCallback


class CallbackList:
    """回调列表"""

    def __init__(self):
        self._callbacks: List[BaseCallback] = []

    def append(self, callback: BaseCallback) -> None:
        self._callbacks.append(callback)

    def init_callback(self, trainer) -> None:
        for cb in self._callbacks:
            cb.init_callback(trainer)

    def on_training_start(self) -> None:
        for cb in self._callbacks:
            cb.on_training_start()

    def on_step(self) -> bool:
        """按序调用所有回调，任一返回 False 则停止训练"""
        for cb in self._callbacks:
            if not cb.on_step():
                return False
        return True

    def on_training_end(self) -> None:
        for cb in self._callbacks:
            cb.on_training_end()

    def __len__(self) -> int:
        return len(self._callbacks)


__all__ = ["CallbackList"]
