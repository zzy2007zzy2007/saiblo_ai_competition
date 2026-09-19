from typing import Any, Dict, Optional

import torch
from loguru import logger


class CheckpointManager:
    """模型检查点管理器 - 保存和加载训练状态"""

    def __init__(self, device: torch.device):
        self._device = device

    def save(
        self,
        filepath: str,
        episode: int,
        policy_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        value_optimizer_state: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        checkpoint = {
            "episode": episode,
            "policy_state_dict": policy_state,
            "optimizer_state_dict": optimizer_state,
            "value_optimizer_state_dict": value_optimizer_state,
            "metrics": metrics or {},
            "config": config or {},
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath} at episode {episode}")

    def load(
        self,
        filepath: str,
        policy,
        optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
    ) -> int:
        checkpoint = torch.load(filepath, map_location=self._device)

        if "policy_state_dict" not in checkpoint:
            raise KeyError(f"Checkpoint {filepath} missing 'policy_state_dict'")
        policy.load_state_dict(checkpoint["policy_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            logger.warning("Checkpoint missing optimizer state, skipping")

        if "value_optimizer_state_dict" in checkpoint:
            value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        else:
            logger.warning("Checkpoint missing value_optimizer state, skipping")

        episode = checkpoint.get("episode", 0)
        logger.info(f"Checkpoint loaded from {filepath}, episode {episode}")
        return episode
