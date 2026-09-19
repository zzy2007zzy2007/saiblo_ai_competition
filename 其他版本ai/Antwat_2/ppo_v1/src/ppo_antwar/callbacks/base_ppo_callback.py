from typing import TYPE_CHECKING, Optional, Any

if TYPE_CHECKING:
    from ..trainer.ppo_trainer import PPOTrainer


class BaseCallback:
    """本地BaseCallback实现 - 兼容openrl接口"""

    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self._trainer_ref = None

    def init_callback(self, trainer: "PPOTrainer") -> None:
        """由PPOTrainer调用，注入训练器引用"""
        self._trainer_ref = trainer

    @property
    def trainer(self) -> "PPOTrainer":
        return self._trainer_ref

    def on_training_start(self) -> None:
        pass

    def on_training_end(self) -> None:
        pass

    def on_step(self) -> bool:
        return True


class BasePPOCallback(BaseCallback):
    """PPO训练器回调基类"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._trainer_ref = None

    def init_callback(self, trainer: "PPOTrainer") -> None:
        """由PPOTrainer调用，注入训练器引用"""
        self._trainer_ref = trainer

    @property
    def trainer(self) -> "PPOTrainer":
        return self._trainer_ref
