import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from .battle_logger import LogLevel


@dataclass
class BaselineBattleConfig:
    """Baseline对战配置"""

    # 对战配置
    max_rounds: int = 512
    n_battles: int = 4  # 并行进程数（先手/后手各一次为固定逻辑）

    # 日志配置
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    log_to_console: bool = True
    enable_timing: bool = True
    timing_warning_threshold: float = 2.0

    # Agent配置
    ppo_checkpoint_path: Optional[str] = None
    baseline_agents: List[str] = field(default_factory=list)
    agent1_name: Optional[str] = None
    agent2_name: Optional[str] = None

    # PPO训练集成配置
    battle_interval: int = 1000
    checkpoint_save_interval: Optional[int] = None
    device: str = "auto"

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'BaselineBattleConfig':
        """从字典创建配置"""
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def load_from_args(self, args):
        """从命令行参数加载配置"""
        if hasattr(args, 'max_rounds') and args.max_rounds is not None:
            self.max_rounds = args.max_rounds
        if hasattr(args, 'n_battles') and args.n_battles is not None:
            self.n_battles = args.n_battles
        if hasattr(args, 'workers') and args.workers is not None:
            self.n_battles = args.workers
        if hasattr(args, 'log_level') and args.log_level is not None:
            self.log_level = args.log_level
        if hasattr(args, 'log_dir') and args.log_dir is not None:
            self.log_dir = args.log_dir
        if hasattr(args, 'log_to_console') and args.log_to_console is not None:
            self.log_to_console = args.log_to_console
        if hasattr(args, 'ppo_checkpoint_path') and args.ppo_checkpoint_path is not None:
            self.ppo_checkpoint_path = args.ppo_checkpoint_path
        if hasattr(args, 'baseline_agents') and args.baseline_agents is not None:
            self.baseline_agents = args.baseline_agents
        if hasattr(args, 'agents') and args.agents is not None and len(args.agents) >= 2:
            self.agent1_name = args.agents[0]
            self.agent2_name = args.agents[1]
        if hasattr(args, 'agent1_name') and args.agent1_name is not None:
            self.agent1_name = args.agent1_name
        if hasattr(args, 'agent2_name') and args.agent2_name is not None:
            self.agent2_name = args.agent2_name
        if hasattr(args, 'device') and args.device is not None:
            self.device = args.device
        if hasattr(args, 'battle_interval') and args.battle_interval is not None:
            self.battle_interval = args.battle_interval

    def get_log_level(self) -> LogLevel:
        """获取日志级别枚举"""
        return LogLevel.from_string(self.log_level)

    def validate(self) -> List[str]:
        """验证配置，返回错误信息列表"""
        errors = []

        if self.n_battles < 1:
            errors.append(f"n_battles must be >= 1, got {self.n_battles}")
            self.n_battles = 4

        if self.max_rounds < 1:
            errors.append(f"max_rounds must be >= 1, got {self.max_rounds}")
            self.max_rounds = 512

        if self.ppo_checkpoint_path and not os.path.exists(self.ppo_checkpoint_path):
            errors.append(f"PPO checkpoint not found: {self.ppo_checkpoint_path}")

        parsed_level = LogLevel.from_string(self.log_level)
        if parsed_level == LogLevel.INFO and self.log_level.upper() not in ['INFO', 'DEBUG', 'WARNING', 'ERROR']:
            errors.append(f"Invalid log_level: {self.log_level}, using INFO")
            self.log_level = "INFO"

        return errors

    def create_logger(self, agent1_name: str, agent2_name: str, path_config) -> 'BattleLogger':
        """便捷方法：基于当前配置创建 logger"""
        from .battle_logger import BattleLogger, LogLevel
        return BattleLogger(
            agent1_name=agent1_name,
            agent2_name=agent2_name,
            log_level=LogLevel.from_string(self.log_level),
            log_to_console=self.log_to_console,
            enable_timing=self.enable_timing,
            path_config=path_config,
            warning_threshold=self.timing_warning_threshold,
        )
