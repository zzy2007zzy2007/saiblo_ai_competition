import math
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class LRScheduleConfig:
    """学习率调度配置"""

    lr_base: float
    lr_vf_base: float
    warmup_episodes: int = 0
    cosine_decay: bool = True


class LRScheduler:
    """学习率调度器 - 支持 warmup + cosine decay + fixed 三种策略"""

    def __init__(self, config: LRScheduleConfig):
        self._config = config

    @classmethod
    def from_ppo_config(cls, ppo_cfg: Dict) -> "LRScheduler":
        return cls(
            LRScheduleConfig(
                lr_base=ppo_cfg.get("lr"),
                lr_vf_base=ppo_cfg.get("lr_vf"),
                warmup_episodes=ppo_cfg.get("lr_warmup_episodes") or 0,
                cosine_decay=ppo_cfg.get("lr_cosine_decay", False),
            )
        )

    def compute(self, episode: int, total_episodes: int) -> Tuple[float, float, str]:
        """计算当前 episode 的策略网络和价值网络学习率"""
        cfg = self._config

        if cfg.warmup_episodes > 0 and episode <= cfg.warmup_episodes:
            ratio = episode / max(cfg.warmup_episodes, 1)
            schedule_type = "warmup"
            lr = cfg.lr_base * ratio
            lr_vf = cfg.lr_vf_base * ratio
            return lr, lr_vf, schedule_type

        if cfg.cosine_decay:
            progress = episode / max(total_episodes, 1)
            factor = (1.0 + math.cos(math.pi * progress)) / 2.0
            lr = cfg.lr_base * factor
            lr_vf = cfg.lr_vf_base * factor
            schedule_type = "cosine"
            return lr, lr_vf, schedule_type

        schedule_type = "fixed"
        return cfg.lr_base, cfg.lr_vf_base, schedule_type

    @staticmethod
    def apply(optimizer, value_optimizer, lr: float, lr_vf: float) -> None:
        """将计算出的学习率应用到优化器"""
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in value_optimizer.param_groups:
            param_group["lr"] = lr_vf
