from typing import Any, List, Tuple
import numpy as np
import torch
from loguru import logger

from .ant_war_agent import AntWarAgent
from ..env.observation import ObservationEncoder
from ..env.action_mask import ActionMaskHandler
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..utils.obs_utils import observation_to_tensors


class GAWarAgent(AntWarAgent):
    """GA 个体驱动的对战 Agent — 语义分段动作版本。

    网络直接输出 (strategy, type_idx, sub_type, position) 四元组，
    通过 ACTION_DECODE_TABLE + 多表位置映射解码为 Operation。
    不再需要 action_mask 输入。
    """

    def __init__(
        self,
        player_id: int,
        network: AntWarPolicyValueNetwork,
        device: torch.device,
    ):
        super().__init__(player_id, name="GA")
        self._network = network
        self._device = device
        self._observation_encoder = ObservationEncoder()
        self._action_mask_handler = ActionMaskHandler()
        self._illegal_count = 0

    def choose_operations(self, state) -> List[Any]:
        """根据当前状态选择操作（确定性策略）。

        流程：
        1. 编码观测（不再包含 action_mask）
        2. 网络推理 → (strategy, type_idx, sub_type, position)
        3. construct_operation() → Operation
        """
        obs = self._observation_encoder.encode(state, self.player_id)
        with np.errstate(invalid="ignore"):
            strategy, type_idx, sub_type, position = self._act(obs)

        op = self._action_mask_handler.construct_operation(
            strategy, type_idx, sub_type, position, state, self.player_id
        )
        if op is not None:
            return [op]

        # 非法动作 → NOOP 惩罚
        self._illegal_count += 1
        logger.warning(
            f"[GAWarAgent] Illegal action "
            f"(strategy={strategy}, type_idx={type_idx}, "
            f"sub_type={sub_type}, position={position}) "
            f"for player={self.player_id}, NOOP penalty applied"
        )
        return []

    def _act(self, observation) -> Tuple[int, int, int, int]:
        """网络推理获取动作四元组（确定性）"""
        tensors = observation_to_tensors(observation, self._device)
        with torch.no_grad():
            strategy, type_idx, sub_type, position = self._network.get_action(
                tensors["board"],
                tensors["global"],
            )
        return strategy, type_idx, sub_type, position

    @property
    def illegal_action_count(self) -> int:
        return self._illegal_count
