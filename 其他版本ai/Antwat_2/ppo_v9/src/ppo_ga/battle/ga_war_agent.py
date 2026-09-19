from typing import Any, List
import numpy as np
import torch
from loguru import logger

from .ant_war_agent import AntWarAgent
from ..env.observation import ObservationEncoder
from ..env.action_mask import ActionMaskHandler
from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..utils.obs_utils import observation_to_tensors


class GAWarAgent(AntWarAgent):
    """GA 个体驱动的对战 Agent。

    与 ppo_v2 的 PPOWarAgent 类似，但直接持有网络实例（而非通过 NeuralAgent 间接引用），
    因为 GA 个体不需要 log_prob/value 等训练元数据。
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
        1. 编码观测
        2. 网络前向推理，获取动作
        3. 转换为 Operation

        Args:
            state: 游戏状态

        Returns:
            Operation 列表
        """
        obs = self._observation_encoder.encode(state, self.player_id)
        with np.errstate(invalid="ignore"):
            action_id = self._act(obs)

        # action_id=0 是 NOOP（不行动），无需转换为 Operation
        if action_id == 0:
            return []

        op = self._action_mask_handler._action_id_to_op(
            action_id, state, self.player_id
        )
        if op is not None:
            return [op]

        # 非法动作回退到随机合法动作
        self._illegal_count += 1
        logger.warning(
            f"[GAWarAgent] Illegal action_id={action_id} for player={self.player_id}, "
            f"falling back to random valid action"
        )
        mask = self._action_mask_handler.get_action_mask(state, self.player_id)
        valid = np.where(mask > 0.5)[0]
        if len(valid) > 0:
            action_id = int(np.random.choice(valid))
            op = self._action_mask_handler._action_id_to_op(
                action_id, state, self.player_id
            )
            return [op] if op is not None else []
        return []

    def _act(self, observation) -> int:
        """网络推理获取动作 ID（确定性）"""
        tensors = observation_to_tensors(observation, self._device)
        with torch.no_grad():
            action_id, _, _, _ = self._network.get_action(
                tensors["board"],
                tensors["global"],
                tensors["action_mask"],
                deterministic=True,
                exploration_epsilon=0.0,
            )
        return action_id

    @property
    def illegal_action_count(self) -> int:
        return self._illegal_count
