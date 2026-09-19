from typing import Any, List

import numpy as np
from loguru import logger

from .ant_war_agent import AntWarAgent


class PPOWarAgent(AntWarAgent):
    """PPO 策略驱动的 Agent"""

    def __init__(
        self,
        player_id: int,
        ppo_agent: Any,
        observation_encoder: Any,
        action_mask_handler: Any,
    ):
        super().__init__(player_id, name="PPO")
        self._ppo_agent = ppo_agent
        self._observation_encoder = observation_encoder
        self._action_mask_handler = action_mask_handler

    def choose_operations(self, state) -> List[Any]:
        obs = self._observation_encoder.encode(state, self.player_id)
        with np.errstate(invalid="ignore"):
            action_id = self._ppo_agent.act(obs)
        op = self._action_mask_handler._action_id_to_op(
            action_id, state, self.player_id
        )
        if op is not None:
            return [op]

        # 非法操作：记录日志，回退到随机合法动作
        logger.warning(
            f"[PPOWarAgent] Illegal action_id={action_id} for player={self.player_id}, "
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


__all__ = ["PPOWarAgent"]
