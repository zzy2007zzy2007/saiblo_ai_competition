import os
from typing import Any, Dict, Optional

import yaml

import torch

from loguru import logger


def _get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    logger.warning(
        "CUDA is not available. Defaulting to CPU. GPU acceleration is not available."
    )
    return "cpu"


_DEFAULT_BOARD_SHAPE = (28, 19, 19)
_DEFAULT_GLOBAL_DIM = 33
_DEFAULT_PLAYER_ID = 0
_DEFAULT_BACKEND_TYPE = "python"
_DEFAULT_PREFER_NATIVE = False

_STRUCTURAL_DEFAULTS: Dict[str, Any] = {
    "system": {"device": _get_default_device()},
    "network": {},
    "env": {
        "player_id": _DEFAULT_PLAYER_ID,
        "backend_type": _DEFAULT_BACKEND_TYPE,
        "prefer_native": _DEFAULT_PREFER_NATIVE,
    },
}

_TYPE_MAPPING: Dict[str, type] = {
    "ppo.lr": float,
    "ppo.lr_vf": float,
    "ppo.gamma": float,
    "ppo.gae_lambda": float,
    "ppo.clip_eps": float,
    "ppo.clip_eps_vf": float,
    "ppo.ent_coef": float,
    "ppo.exploration_epsilon": float,
    "ppo.vf_coef": float,
    "ppo.max_grad_norm": float,
    "ppo.max_grad_norm_vf": float,
    "ppo.ppo_epochs": int,
    "ppo.batch_size": int,
    "ppo.lr_cosine_decay": bool,
    "ppo.enable_auxiliary": bool,
    "training.total_episodes": int,
    "training.save_interval": int,
    "training.opponent_update_interval": int,
    "training.n_envs": int,
    "training.battle_interval": int,
    "training.n_battles": int,
    "training.max_steps_per_episode": int,
    "selfplay.opponent_pool_size": int,
    "selfplay.min_opponent_games": int,
    "selfplay.exploit_prob": float,
    "network.hidden_dim": int,
}


class PPORLConfigParser:
    """配置解析器 - 封装配置的加载和合并逻辑"""

    @staticmethod
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def merge_yaml_and_args(
        yaml_path: str, cli_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = PPORLConfigParser.load_yaml(yaml_path)
        config = PPORLConfigParser._merge_structural_defaults(config)
        if cli_overrides:
            cli_overrides = PPORLConfigParser._filter_none_values(cli_overrides)
            cli_overrides = PPORLConfigParser._validate_types(cli_overrides)
            if cli_overrides:
                config = PPORLConfigParser._deep_merge(config, cli_overrides)
        return config

    @staticmethod
    def _merge_structural_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        for section, values in _STRUCTURAL_DEFAULTS.items():
            if section not in config:
                config[section] = {}
            for key, val in values.items():
                if key not in config[section]:
                    config[section][key] = val
        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = PPORLConfigParser._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _filter_none_values(data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if value is None:
                continue
            if isinstance(value, dict):
                filtered = PPORLConfigParser._filter_none_values(value)
                if filtered:
                    result[key] = filtered
            else:
                result[key] = value
        return result

    @staticmethod
    def _validate_types(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            current_path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                validated = PPORLConfigParser._validate_types(value, current_path)
                if validated:
                    result[key] = validated
            elif current_path in _TYPE_MAPPING:
                expected_type = _TYPE_MAPPING[current_path]
                try:
                    if expected_type is float:
                        if isinstance(value, (int, float)):
                            result[key] = float(value)
                        elif isinstance(value, str):
                            try:
                                result[key] = float(value)
                            except ValueError:
                                logger.warning(
                                    f"类型转换失败: {current_path} 期望 float, 但值 '{value}' 无法解析, 跳过覆盖"
                                )
                                continue
                        else:
                            result[key] = value
                    elif expected_type is int:
                        if isinstance(value, str):
                            result[key] = int(value)
                        else:
                            result[key] = int(value)
                    elif expected_type is bool:
                        if isinstance(value, str):
                            lowered = value.strip().lower()
                            if lowered in ("true", "1"):
                                result[key] = True
                            elif lowered in ("false", "0"):
                                result[key] = False
                            else:
                                logger.warning(
                                    f"类型转换失败: {current_path} 期望 bool, 但值不可解析 '{value}', 跳过覆盖"
                                )
                                continue
                        else:
                            result[key] = bool(value)
                    else:
                        result[key] = value
                except (ValueError, TypeError):
                    logger.warning(
                        f"类型转换失败: {current_path} 期望 {expected_type.__name__}, 但值为 '{value}', 跳过覆盖"
                    )
                    continue
            else:
                result[key] = value
        return result


def create_ppo_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建 PPO 配置的便捷入口

    Args:
        yaml_path: YAML 配置文件路径，默认为 src/ppo_antwar/configs/ppo_antwar.yaml
        cli_overrides: CLI 覆盖参数字典

    Returns:
        完整的配置字典
    """
    if yaml_path is None:
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "ppo_antwar.yaml"
        )

    config = PPORLConfigParser.merge_yaml_and_args(yaml_path, cli_overrides)

    device = config["system"]["device"]
    if device == "cpu":
        logger.warning("Training device is CPU. GPU acceleration is not available.")
    elif "cuda" in device and not torch.cuda.is_available():
        logger.warning(
            f"Configured device is '{device}' but CUDA is not available. Falling back to CPU."
        )

    return config
