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


_STRUCTURAL_DEFAULTS: Dict[str, Any] = {
    "system": {"device": _get_default_device()},
    "network": {"hidden_dim": 256},
    "evolution": {
        "max_generations": 200,
        "population_size": 80,
        "gen1_population_size": None,   # None=使用 population_size
        "num_seeds": 8,
        "elitism_count": 2,
        "convergence_window": 20,
        "convergence_threshold": 5.0,
    },
    "crossover": {"method": "layer_wise", "rate": 0.8},
    "mutation": {"method": "gaussian", "rate": 0.1, "scale": 0.03, "decay": 0.99},
    "elite_dedup": {
        "enabled": True,
        "hard_threshold": 0.95,
        "soft_threshold": 0.85,
        "soft_max_per_family": 2,
    },
    "pruning": {
        "enabled": True,
        "n_battle": 2,
        "max_workers": 12,
        "opponent": "NoviceAI",
    },
    "touchstone": {
        "M": 16,
        "max_seeds": 3,
        "n_battle": 1,
        "max_candidate_fails": 12,
        "max_workers": 12,
    },
    "round_robin": {
        "top_n": 24,
        "n_battles": 4,
        "max_workers": 12,
    },
    "baseline_battle": {"enabled": True, "interval": 1, "n_battles": 12, "agents": ["BasicRandomAI", "BasicTowerAI", "MediumRuleAI"]},
    "env": {"player_id": 0, "backend_type": "python", "max_steps": 512},
    "logging": {"log_level": "INFO", "save_interval": 10, "tensorboard": True},
}

_TYPE_MAPPING: Dict[str, type] = {
    "evolution.max_generations": int,
    "evolution.population_size": int,
    "evolution.num_seeds": int,
    "evolution.elitism_count": int,
    "evolution.convergence_window": int,
    "evolution.convergence_threshold": float,
    "crossover.rate": float,
    "mutation.rate": float,
    "mutation.scale": float,
    "mutation.decay": float,
    "touchstone.M": int,
    "touchstone.max_seeds": int,
    "touchstone.n_battle": int,
    "touchstone.max_candidate_fails": int,
    "touchstone.max_workers": int,
    "pruning.enabled": bool,
    "pruning.n_battle": int,
    "pruning.max_workers": int,
    "round_robin.top_n": int,
    "round_robin.n_battles": int,
    "round_robin.max_workers": int,
    "baseline_battle.enabled": bool,
    "baseline_battle.interval": int,
    "baseline_battle.n_battles": int,
    "network.hidden_dim": int,

    "env.max_steps": int,
    "logging.save_interval": int,
    "logging.tensorboard": bool,
}


class GAConfigParser:
    """配置解析器 - 封装 GA 配置的加载和合并逻辑"""

    @staticmethod
    def load_yaml(yaml_path: str) -> Dict[str, Any]:
        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def merge_yaml_and_args(
        yaml_path: str, cli_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config = GAConfigParser.load_yaml(yaml_path)
        config = GAConfigParser._merge_structural_defaults(config)
        if cli_overrides:
            cli_overrides = GAConfigParser._filter_none_values(cli_overrides)
            cli_overrides = GAConfigParser._validate_types(cli_overrides)
            if cli_overrides:
                config = GAConfigParser._deep_merge(config, cli_overrides)
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
                result[key] = GAConfigParser._deep_merge(result[key], value)
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
                filtered = GAConfigParser._filter_none_values(value)
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
                validated = GAConfigParser._validate_types(value, current_path)
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


def create_ga_config(
    yaml_path: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建 GA 配置的便捷入口

    Args:
        yaml_path: YAML 配置文件路径，默认为 configs/ga_evo.yaml
        cli_overrides: CLI 覆盖参数字典

    Returns:
        完整的配置字典
    """
    if yaml_path is None:
        yaml_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "configs", "ga_evo_v10.yaml"
        )

    config = GAConfigParser.merge_yaml_and_args(yaml_path, cli_overrides)

    device = config["system"]["device"]
    if device == "cpu":
        logger.warning("Training device is CPU. GPU acceleration is not available.")
    elif "cuda" in device and not torch.cuda.is_available():
        logger.warning(
            f"Configured device is '{device}' but CUDA is not available. Falling back to CPU."
        )

    return config
