from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BaselineBattleConfig:
    """对战配置 - 管理对战评估的所有可调参数（与 ppo_v2 保持一致）"""
    max_rounds: int
    n_battles: int
    max_workers: int = 12
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    log_to_console: bool = True
    enable_timing: bool = True
    timing_warning_threshold: float = 2.0
    ppo_checkpoint_path: Optional[str] = None
    baseline_agents: List[str] = field(default_factory=list)
    device: str = "auto"


@dataclass
class GABattleConfig(BaselineBattleConfig):
    """GA 对战配置 - 扩展 BaselineBattleConfig，增加 GA 特有参数"""
    hidden_dim: int = 256
    enable_auxiliary: bool = True
