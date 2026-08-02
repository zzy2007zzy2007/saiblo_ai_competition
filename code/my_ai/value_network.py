"""Independent value network for board-state evaluation.

Architecture:
  board(28,19,19) → Conv7×7 + 6×ResBlock(64) → GAP → board_emb(64)
  stats(42)       → MLP → stats_emb(64)
  state_emb = concat(board_emb, stats_emb) = 128
      └── Value head: Linear(128→64) + ReLU + Linear(64→1) → bare scalar

Output: scalar ∈ ℝ (no Tanh/sigmoid constraint).
  Positive → board favours current player (we lead).
  Negative → board favours opponent (we trail).

Training target: (hp_us - hp_opp) / 20.0
  Normalised HP difference; roughly in [-1, 1].
  MSE loss; bare Linear output so the network learns its own scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_ai.network import ResBlock


class ValueNetwork(nn.Module):
    """Value network that estimates board-state value from (board, stats).

    Reuses the same encoder design as AntWarNetwork (Conv7×7 + 6×ResBlock +
    GAP), but discards all policy-related heads.  The output is a bare scalar
    (no Tanh) for unbounded value regression.
    """

    BOARD_CHANNELS = 28
    BOARD_SIZE = 19
    STATS_DIM = 42

    def __init__(self, latent_dim: int = 64, num_resblocks: int = 6):
        super().__init__()
        self.LATENT_DIM = latent_dim

        # Board encoder (same as AntWarNetwork)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.BOARD_CHANNELS, self.LATENT_DIM, kernel_size=7,
                      padding=3, bias=False),
            nn.BatchNorm2d(self.LATENT_DIM),
            nn.ReLU(),
        )
        self.resblocks = nn.ModuleList([
            ResBlock(self.LATENT_DIM) for _ in range(num_resblocks)
        ])

        # Stats encoder
        self.stats_mlp = nn.Sequential(
            nn.Linear(self.STATS_DIM, self.LATENT_DIM),
            nn.ReLU(),
        )

        # Value head — two-layer MLP, bare Linear output (no Tanh)
        self.value_head = nn.Sequential(
            nn.Linear(self.LATENT_DIM * 2, self.LATENT_DIM),  # state_emb = 128
            nn.ReLU(),
            nn.Linear(self.LATENT_DIM, 1),   # bare scalar ∈ ℝ
        )

    def forward(self, board: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            board: (B, 28, 19, 19)
            stats: (B, 42)

        Returns:
            (B,) scalar tensor — estimated board value.
        """
        x = self.initial_conv(board)                 # (B, 64, 19, 19)
        for block in self.resblocks:
            x = block(x)                              # (B, 64, 19, 19)

        board_emb = x.mean(dim=[2, 3])                # (B, 64)  GAP
        stats_emb = self.stats_mlp(stats)              # (B, 64)
        state_emb = torch.cat([board_emb, stats_emb], dim=1)  # (B, 128)

        return self.value_head(state_emb).squeeze(-1)  # (B,)

    def get_parameters_as_vector(self) -> np.ndarray:
        """Flatten all parameters into a single vector (for saving/loading)."""
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(params)

    def set_parameters_from_vector(self, vec: np.ndarray) -> None:
        """Restore parameters from a flattened vector."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(
                torch.from_numpy(vec[idx: idx + size]).view(p.shape)
            )
            idx += size

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def numel(self) -> int:
        return self.count_parameters()


def create_value_network(latent_dim: int = 64,
                         num_resblocks: int = 6) -> ValueNetwork:
    """Factory function for the full-value network (random init).

    Args:
        latent_dim:   latent channel count (default 64).
        num_resblocks: number of ResBlocks (default 6).

    Returns:
        A randomly initialised ValueNetwork.
    """
    return ValueNetwork(latent_dim=latent_dim, num_resblocks=num_resblocks)


def create_small_value_network() -> ValueNetwork:
    """Smaller value network (~97K parameters, 2 ResBlocks, 32 latent dim)."""
    return ValueNetwork(latent_dim=32, num_resblocks=2)


if __name__ == "__main__":
    # Quick smoke test
    model = create_value_network()
    board = torch.randn(1, 28, 19, 19)
    stats = torch.randn(1, 42)
    out = model(board, stats)
    print(f"ValueNetwork parameters: {model.count_parameters():,}")
    print(f"Output shape: {out.shape}  (expected [1])")
    print(f"Output value: {out.item():.4f}  (bare Linear, no Tanh)")

    small = create_small_value_network()
    print(f"Small ValueNetwork parameters: {small.count_parameters():,}")
    out_s = small(board, stats)
    print(f"Small output: {out_s.item():.4f}")
