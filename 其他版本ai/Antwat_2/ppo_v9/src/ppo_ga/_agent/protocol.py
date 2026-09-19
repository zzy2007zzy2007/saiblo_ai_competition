"""Agent 协议定义 - 所有 Agent 类型的抽象基类。"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class Agent(ABC):
    """任何能在环境中做决策的 Agent。"""

    @abstractmethod
    def act(self, observation: Dict[str, np.ndarray]) -> int:
        """根据观测返回动作 ID。

        Args:
            observation: 环境观测，包含 board (28,19,19), global (33,),
                          action_mask (N,) 以及可能的辅助字段。

        Returns:
            选中的动作 ID (int)。
        """
        ...


class RecordingAgent(Agent):
    """能提供训练元数据（log_prob, value）的 Agent。

    相比 Agent.act()，act_record() 额外返回对数概率和状态价值，
    供 PPO 更新时计算重要性采样比率和价值损失。
    """

    @abstractmethod
    def act_record(
        self, observation: Dict[str, np.ndarray]
    ) -> Tuple[int, float, float]:
        """根据观测返回动作及训练元数据。

        Args:
            observation: 环境观测，格式同 Agent.act()。

        Returns:
            (action_id, log_prob, value) 三元组：
            - action_id: int, 选中的动作 ID
            - log_prob:  float, 选中动作的对数概率，是重要性采样比率的分子分母之一
            - value:     float, Critic 输出的状态价值 V(s)，用于 GAE 计算和 value loss
        """
        ...
