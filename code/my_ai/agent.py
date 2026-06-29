"""AI Agent that uses the neural network model for decision making.

Integrates with the game SDK via BaseAgent interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the Ant-Game root and code/ are in the Python path
_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import torch
import numpy as np

from AI.common import BaseAgent
from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle
from SDK.utils.features import FeatureExtractor

from my_ai.network import AntWarNetwork, create_zero_model
from my_ai.decoder import decode_network_output


class NeuralAgent(BaseAgent):
    """Agent that uses AntWarNetwork for decision making."""

    def __init__(
        self,
        model: AntWarNetwork | None = None,
        seed: int | None = None,
        max_actions: int = 96,
    ):
        super().__init__(seed=seed, max_actions=max_actions)
        self.max_actions = max_actions
        self.model = model or create_zero_model()
        self.model.eval()  # inference mode
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)

    def set_model(self, model: AntWarNetwork) -> None:
        """Replace the model (used by ES to update weights)."""
        self.model = model
        self.model.eval()

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        """Use neural network to select actions, then wrap as ActionBundle."""
        operations = self._choose_operations(state, player)
        # Wrap as ActionBundle
        score = 0.0
        if operations:
            # Simple heuristic: more complex bundles get higher score
            # (so the agent prefers doing things over noop)
            score = len(operations) * 10.0 + sum(
                5.0 if op.op_type != 11 else 0.0 for op in operations  # Bonus for non-build
            )
        return ActionBundle(
            name="neural",
            operations=tuple(operations),
            score=score,
            tags=("neural",),
        )

    def _choose_operations(self, state: BackendState, player: int) -> list:
        """Run network forward and decode into operations."""
        # Prepare input
        obs = self.feature_extractor.encode_observation(state, player, np.zeros(self.max_actions))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()  # (1, 28, 19, 19)
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()  # (1, N)

        # Network forward
        with torch.no_grad():
            output = self.model(board, stats)

        # Decode
        operations = decode_network_output(output, state, player)
        return operations
