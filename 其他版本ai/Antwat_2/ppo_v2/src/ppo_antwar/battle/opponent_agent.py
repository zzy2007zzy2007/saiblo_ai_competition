from typing import Any

import torch

from ..network.ant_war_policy_value_network import AntWarPolicyValueNetwork
from ..trainer.neural_agent import NeuralAgent


class OpponentAgent:
    """统一对手Agent接口 - 从检查点加载模型，提供标准的 act(obs) 接口。"""

    def __init__(self, agent: NeuralAgent):
        self._agent = agent

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device,
        hidden_dim: int = 256,
        enable_auxiliary: bool = True,
    ) -> "OpponentAgent":
        policy = AntWarPolicyValueNetwork(
            hidden_dim=hidden_dim,
            enable_auxiliary=enable_auxiliary,
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        return cls(NeuralAgent(policy=policy, device=device, exploration_epsilon=0.0))

    def act(self, observation: Any) -> int:
        return self._agent.act(observation)
