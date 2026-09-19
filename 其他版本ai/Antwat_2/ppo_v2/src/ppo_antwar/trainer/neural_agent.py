"""NeuralAgent - 基于神经网络的 Agent，替代 PPOAgent。

职责：
- 所有神经网络推理的统一封装
- act() 仅做决策（返回 action_id）
- act_record() 返回 (action, log_prob, value)，含 NaN 保护
- 不绑定 PPO 训练逻辑
"""

from typing import Dict, Tuple

import numpy as np
import torch
from loguru import logger

from .._agent.protocol import RecordingAgent
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork


class NeuralAgent(RecordingAgent):
    """基于神经网络的 Agent。

    所有 inference 统一走 _forward()，act() 和 act_record() 只是返回值取舍。
    包含 NaN 保护。
    """

    def __init__(
        self,
        policy: AntWarPolicyValueNetwork,
        device: torch.device,
        exploration_epsilon: float,
    ):
        self._policy = policy
        self._device = device
        self._exploration_epsilon = exploration_epsilon

    def act(self, observation: Dict[str, np.ndarray]) -> int:
        action_id, _, _ = self._forward(observation)
        return action_id

    def act_record(
        self, observation: Dict[str, np.ndarray]
    ) -> Tuple[int, float, float]:
        return self._forward(observation)

    def get_value(self, observation: Dict[str, np.ndarray]) -> float:
        """获取状态价值（仅运行 value encoder，跳过 policy 前向）。"""
        from ..utils.obs_utils import observation_to_tensors

        tensors = observation_to_tensors(observation, self._device)
        with torch.no_grad():
            value = self._policy.get_value_only(
                tensors["board"], tensors["global"]
            )
        return value.item()

    def _forward(
        self, observation: Dict[str, np.ndarray]
    ) -> Tuple[int, float, float]:
        """统一推理入口。含 NaN 保护。"""
        from ..utils.obs_utils import observation_to_tensors

        with torch.no_grad():
            tensors = observation_to_tensors(observation, self._device)

            action_id, log_prob, value, _ = self._policy.get_action(
                tensors["board"],
                tensors["global"],
                tensors["action_mask"],
                deterministic=False,
                exploration_epsilon=self._exploration_epsilon,
            )

        # NaN 保护（统一放在 _forward 中，所有调用方自动受益）
        has_nan = False
        if log_prob.numel() > 0 and torch.isnan(log_prob).any().item():
            has_nan = True
        if value.numel() > 0 and torch.isnan(value).any().item():
            has_nan = True
        if has_nan:
            logger.warning("NaN detected in action selection, using fallback")
            log_prob = torch.tensor(0.0, device=self._device)
            value = torch.tensor(0.0, device=self._device)

        return (
            action_id,
            log_prob.item() if log_prob.numel() > 0 else 0.0,
            value.item() if value.numel() > 0 else 0.0,
        )
