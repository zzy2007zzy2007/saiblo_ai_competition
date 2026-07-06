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
from my_ai.gating_network import GatingNetwork


class NeuralAgent(BaseAgent):
    """Agent that uses AntWarNetwork for decision making."""

    def __init__(
        self,
        model: AntWarNetwork | None = None,
        seed: int | None = None,
        max_actions: int = 96,
        *,
        allowed_classes: list[int] | None = None,
    ):
        super().__init__(seed=seed, max_actions=max_actions)
        self.max_actions = max_actions
        self.model = model or create_zero_model()
        self.model.eval()  # inference mode
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)
        self.last_output = None
        self.allowed_classes = allowed_classes

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

        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        board = board.to(device)
        stats = stats.to(device)

        # Network forward
        with torch.no_grad():
            output = self.model(board, stats)
        self.last_output = output

        # Decode (with optional class restriction)
        operations = decode_network_output(output, state, player,
                                           allowed_classes=self.allowed_classes)
        return operations


class MultiExpertAgent(BaseAgent):
    """Agent that uses a gating network to select among expert models.

    Each expert is a NeuralAgent with its own ``allowed_classes`` restriction.
    The gating network takes the average state embedding across all experts
    and selects which expert to activate for the current turn.
    """

    def __init__(
        self,
        experts: list[NeuralAgent],
        gate: GatingNetwork,
        seed: int | None = None,
        max_actions: int = 96,
    ):
        super().__init__(seed=seed, max_actions=max_actions)
        self.experts = experts
        self.gate = gate
        self.gate.eval()
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)
        self.last_output = None

    def set_experts(self, experts: list[NeuralAgent]) -> None:
        """Replace the expert list (used by ES to load trained experts)."""
        self.experts = experts

    def set_gate(self, gate: GatingNetwork) -> None:
        """Replace the gating network."""
        self.gate = gate
        self.gate.eval()

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        """Use gating network to select expert, then wrap as ActionBundle."""
        operations = self._choose_operations(state, player)
        score = 0.0
        if operations:
            score = len(operations) * 10.0 + sum(
                5.0 if op.op_type != 11 else 0.0 for op in operations
            )
        return ActionBundle(
            name="multi_expert",
            operations=tuple(operations),
            score=score,
            tags=("multi_expert",),
        )

    def _choose_operations(self, state: BackendState, player: int) -> list:
        """Run all experts forward, gate to select one, decode its output."""
        obs = self.feature_extractor.encode_observation(
            state, player, np.zeros(self.max_actions))
        board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
        stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()

        device = next(self.experts[0].model.parameters()).device
        board = board.to(device)
        stats = stats.to(device)

        # Run all experts forward, collect outputs and state embeddings
        state_embs: list[torch.Tensor] = []
        expert_outputs: list[dict] = []
        with torch.no_grad():
            for expert in self.experts:
                out = expert.model(board, stats)
                expert_outputs.append(out)
                state_embs.append(out["state_emb"])

        # Gating: average state embeddings across all experts
        avg_emb = torch.mean(torch.stack(state_embs), dim=0)  # (1, 128)
        gate_logits = self.gate(avg_emb)  # (1, num_experts)
        expert_idx = gate_logits.argmax(dim=1).item()

        # Decode selected expert's output
        selected_out = expert_outputs[expert_idx]
        self.last_output = selected_out

        operations = decode_network_output(
            selected_out, state, player,
            allowed_classes=self.experts[expert_idx].allowed_classes,
        )
        return operations
