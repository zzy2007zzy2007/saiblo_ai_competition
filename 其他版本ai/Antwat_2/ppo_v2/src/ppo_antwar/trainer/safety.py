from typing import Any, Dict, Optional

import copy

import torch
from loguru import logger

_CLEAN_NAN_REPLACEMENT = 0.0
_CLEAN_POSINF_REPLACEMENT = 1e4
_CLEAN_NEGINF_REPLACEMENT = -1e4


def check_tensor_nan(tensor: torch.Tensor, name: str = "") -> bool:
    """检测张量中是否包含 NaN 或 Inf 值"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        logger.warning(
            f"NaN/Inf detected in tensor '{name}': nan={has_nan}, inf={has_inf}"
        )
        return True
    return False


def clean_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """清理张量中的 NaN/Inf 值"""
    return torch.nan_to_num(tensor, nan=0.0, posinf=1e4, neginf=-1e4)


class NanRecoveryHandler:
    """NaN 检测与恢复管理器

    职责：
    - 跟踪 NaN 发生次数
    - 维护最近一次健康检查点
    - 在 NaN 超过阈值时触发恢复（回滚权重 + 停止训练）
    """

    def __init__(self, nan_threshold: int):
        self._nan_count: int = 0
        self._nan_threshold: int = nan_threshold
        self._last_good_policy_state: Optional[Dict[str, torch.Tensor]] = None
        self._last_good_optimizer_state: Optional[Dict[str, Any]] = None
        self._last_good_value_optimizer_state: Optional[Dict[str, Any]] = None

    @property
    def nan_count(self) -> int:
        return self._nan_count

    def save_good_state(
        self,
        policy_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Any],
        value_optimizer_state: Dict[str, Any],
    ) -> None:
        self._last_good_policy_state = {k: v.clone() for k, v in policy_state.items()}
        self._last_good_optimizer_state = copy.deepcopy(optimizer_state)
        self._last_good_value_optimizer_state = copy.deepcopy(value_optimizer_state)

    def on_nan_detected(
        self,
        episode: int,
        policy,
        optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
    ) -> bool:
        """NaN 恢复机制

        Returns:
            True 表示继续训练，False 表示应停止训练
        """
        self._nan_count += 1
        logger.warning(
            f"NaN detected at episode {episode}, count: {self._nan_count}/{self._nan_threshold}"
        )

        if self._nan_count >= self._nan_threshold:
            logger.error(
                f"NaN count exceeded threshold {self._nan_threshold}, stopping training"
            )
            return False

        if self._last_good_policy_state is not None:
            logger.info("Restoring from last good checkpoint")
            policy.load_state_dict(self._last_good_policy_state)
            optimizer.load_state_dict(self._last_good_optimizer_state)
            value_optimizer.load_state_dict(self._last_good_value_optimizer_state)

        return True
