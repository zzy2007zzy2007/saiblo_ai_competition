"""PPO配置解析模块 - 集成OpenRL配置系统"""

from openrl.configs.config import create_config_parser as _create_config_parser
from easydict import EasyDict
from typing import Any, Dict, Optional
import yaml


class PPORLConfigParser:
    """PPO训练配置解析器 - 封装OpenRL配置系统"""

    def __init__(self):
        self._parser = None
        self._cfg = None

    def parse_args(self, args=None):
        """解析命令行参数"""
        self._parser = _create_config_parser()
        self._cfg = self._parser.parse_args(args)
        return self

    def load_yaml(self, yaml_path: str) -> EasyDict:
        """从YAML文件加载配置"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return EasyDict(config_dict)

    def merge_yaml_and_args(self, yaml_path: str, args=None) -> EasyDict:
        """合并YAML配置和命令行参数"""
        config = self.load_yaml(yaml_path)

        if args:
            self._parser = _create_config_parser()
            cli_cfg = self._parser.parse_args(args)

            for key in dir(cli_cfg):
                if not key.startswith('_') and hasattr(cli_cfg, key):
                    value = getattr(cli_cfg, key)
                    if value is not None:
                        config = self._merge_config(config, {key: value})

        return EasyDict(config)

    def _merge_config(self, base: Dict, override: Dict) -> Dict:
        """深度合并配置字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    @property
    def cfg(self):
        return self._cfg


# 仅含网络输入/输出的结构性约束，可调超参需在 YAML 中显式定义
# `network` 节仅含 `board_shape` 和 `global_dim` 决定观测维度，
# 可调超参（`hidden_dim`、`action_dim`）需在 YAML 中显式定义
def _get_default_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


_STRUCTURAL_DEFAULTS = {
    "system": {"device": _get_default_device()},
    "network": {
        "board_shape": [28, 19, 19],
        "global_dim": 33,
    },
    "env": {"player_id": 0, "backend_type": "python", "prefer_native": False},
}


def _filter_none(d: dict) -> dict:
    """递归过滤 dict 中的 None 值。

    注意：会递归移除所有值为 None 的嵌套键。若整个子树均为 None（如
    {"training": {"warmup": {"a": None, "b": None}}}），整个子树将不产生任何合并操作。
    调用方应通过 `if cli_overrides:` 守卫确保过滤后的空 dict 不会意外覆盖已有配置。
    """
    result = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            filtered = _filter_none(v)
            if filtered:
                result[k] = filtered
        else:
            result[k] = v
    return result


def _deep_merge_overrides(config: EasyDict, overrides: dict):
    """递归深度合并 cli_overrides 到 config。

    对 overrides 中每个 key：
      - 值为 None → 跳过（由 _filter_none 预先过滤，此处为防御）
      - 值不是 dict → 直接覆盖 config 的同名 key
      - 值是 dict 且 config 中对应 key 也是 dict → 递归合并
      - 值是 dict 但 config 中对应 key 不存在或不是 dict → 整体覆盖

    设计意图：允许通过 cli_overrides 创建 YAML 中不存在的新 section（如单元测试中的自定义实验参数）。
    此行为与 Python dict 的 update 语义一致，是有意设计，非边界条件。
    """
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict):
            if key not in config:
                config[key] = EasyDict()
            if isinstance(config[key], dict):
                _deep_merge_overrides(config[key], value)
            else:
                config[key] = EasyDict(value)
        else:
            config[key] = value


def create_ppo_config(
    yaml_path: str,
    args: Optional[list] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> EasyDict:
    """创建PPO配置的便捷函数

    Args:
        yaml_path: YAML配置文件路径（必需）
        args: 命令行参数列表（预留接口，当前无项目内调用方使用）
        cli_overrides: 命令行参数字典（优先级最高）

    Returns:
        EasyDict 配置对象
    """
    parser = PPORLConfigParser()
    config = parser.merge_yaml_and_args(yaml_path, args)

    # 合并结构性兜底（YAML 缺失时补充网络结构等不可推导的常量）
    for section, defaults in _STRUCTURAL_DEFAULTS.items():
        if section not in config:
            config[section] = EasyDict(defaults)
        elif isinstance(config[section], dict):
            for k, v in defaults.items():
                if k not in config[section]:
                    config[section][k] = v

    # 深度合并 CLI 覆盖值（优先级最高）
    if cli_overrides:
        filtered = _filter_none(cli_overrides)
        _deep_merge_overrides(config, filtered)

    return config
