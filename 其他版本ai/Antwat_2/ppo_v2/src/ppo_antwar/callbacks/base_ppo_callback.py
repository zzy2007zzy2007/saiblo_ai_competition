from typing import Any, Callable, Dict, Optional

from .base_callback import BaseCallback


class BasePPOCallback(BaseCallback):
    """PPO 训练专用回调基类

    子类通过重写 _on_step 实现自定义逻辑。
    """

    def __init__(self):
        self._trainer_ref: Optional[Any] = None
        self._last_metrics_provider: Optional[Callable[[], Optional[Dict]]] = None

    @property
    def trainer(self) -> Any:
        return self._trainer_ref

    def init_callback(self, trainer: Any) -> None:
        self._trainer_ref = trainer
        # 若 trainer 实现了 get_last_metrics()，注入访问方法
        if hasattr(trainer, "get_last_metrics"):
            self._last_metrics_provider = trainer.get_last_metrics

    def get_last_metrics(self) -> Optional[Dict]:
        """获取最近一次 PPO update 的训练指标字典"""
        if self._last_metrics_provider is not None:
            return self._last_metrics_provider()
        # 向后兼容（在 Trainer 未实现接口时降级）
        if self._trainer_ref is not None:
            return getattr(self._trainer_ref, "_last_metrics", None)
        return None

    def on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_step(self) -> bool:
        return self._on_step()

    def _on_step(self) -> bool:
        return True


__all__ = ["BasePPOCallback"]
