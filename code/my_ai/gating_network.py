"""Gating network for Multi-Expert Agent.

Selects which expert to use given the game state embedding.
Input: state_emb (128-dim, from the shared encoder's board_emb + stats_emb)
Output: expert logits (num_experts-dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingNetwork(nn.Module):
    """Small network that selects which expert to activate.

    Architecture:
        state_emb(128) → Linear(128→64) → ReLU → Linear(64→32) → ReLU → Linear(32→N)

    Total params: 128*64 + 64*32 + 32*N ≈ 10K + 32*N
    For N=4: ~10K params (tiny, easily optimized by ES).
    """

    def __init__(self, num_experts: int = 4, input_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )

    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state_emb: (B, 128) — board embedding + stats embedding

        Returns:
            expert_logits: (B, num_experts) — raw logits (no softmax)
        """
        return self.net(state_emb)

    def get_parameters_as_vector(self) -> "np.ndarray":
        """Flatten all parameters into a single vector (for ES)."""
        import numpy as np
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(params)

    def set_parameters_from_vector(self, vec: "np.ndarray") -> None:
        """Restore parameters from a flattened vector."""
        import numpy as np
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(vec[idx: idx + size]).view(p.shape))
            idx += size

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
