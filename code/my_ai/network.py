"""Neural network model for Ant-Game AI.

Architecture (AlphaZero-style):
  board(28,19,19) → Conv7×7 + ResBlock×6 → spatial_feat(64,19,19)
      ├── Conv1×1, 64→23 → action_map(23,19,19)
      └── GAP → board_emb(64)
  stats(~22) → MLP → stats_emb(64)
  state_emb = concat(board_emb, stats_emb) = 128
      ├── Value Head → scalar [-1, 1]
      └── Policy Head → 3 × 23 class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """Residual block with two Conv3×3 layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x


class AntWarNetwork(nn.Module):
    """Full neural network for Ant-Game.

    NUM_CLASSES = 24:
      0-15  → tower actions (build/upgrade/downgrade)
      16    → downgrade tower
      17-20 → super weapons
      21-22 → base upgrades
      23    → HOLD (do nothing this turn)

    When ``single_head=True``, only one policy head is created (head1),
    reducing parameters by ~5.9K. This is useful for early-stage ES training
    to avoid conflicting behavior between multiple heads.
    """

    NUM_CLASSES = 24  # 0-22 action classes + 23 = HOLD
    BOARD_CHANNELS = 28
    BOARD_SIZE = 19
    STATS_DIM = 42
    LATENT_DIM = 64

    def __init__(self, num_resblocks: int = 6, single_head: bool = False):
        super().__init__()
        self.num_resblocks = num_resblocks
        self.single_head = single_head

        # Board encoder
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.BOARD_CHANNELS, self.LATENT_DIM, kernel_size=7, padding=3, bias=False),
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

        # Policy head - spatial
        self.action_map_conv = nn.Conv2d(self.LATENT_DIM, self.NUM_CLASSES, kernel_size=1)

        # Policy head - class
        self.policy_base = nn.Sequential(
            nn.Linear(self.LATENT_DIM, self.LATENT_DIM),
            nn.ReLU(),
        )
        self.policy_head1 = nn.Linear(self.LATENT_DIM, self.NUM_CLASSES)
        if not single_head:
            self.policy_head2 = nn.Linear(self.LATENT_DIM, self.NUM_CLASSES)
            self.policy_head3 = nn.Linear(self.LATENT_DIM, self.NUM_CLASSES)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.LATENT_DIM * 2, self.LATENT_DIM),  # state_emb = board_emb(64) + stats_emb(64)
            nn.ReLU(),
            nn.Linear(self.LATENT_DIM, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor, stats: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            board: (B, 28, 19, 19)
            stats: (B, 22)
        Returns:
            dict with keys:
              - action_map: (B, 23, 19, 19)
              - head1_logits: (B, 23)
              - head2_logits, head3_logits: (B, 23) — only if single_head=False
              - value: (B, 1)
        """
        # Board encoder
        x = self.initial_conv(board)  # (B, 64, 19, 19)
        for block in self.resblocks:
            x = block(x)  # (B, 64, 19, 19)
        spatial_feat = x  # keep for action map

        # Global average pooling → board embedding
        board_emb = spatial_feat.mean(dim=[2, 3])  # (B, 64)

        # Stats encoder
        stats_emb = self.stats_mlp(stats)  # (B, 64)

        # State embedding
        state_emb = torch.cat([board_emb, stats_emb], dim=1)  # (B, 128)

        # --- Policy head ---
        # Spatial: action map (shared)
        action_map = self.action_map_conv(spatial_feat)  # (B, 23, 19, 19)

        # Class heads
        policy_base = self.policy_base(board_emb)  # (B, 64)
        head1_logits = self.policy_head1(policy_base)  # (B, 23)
        result: dict[str, torch.Tensor] = {
            "action_map": action_map,
            "head1_logits": head1_logits,
            "value": self.value_head(state_emb),  # (B, 1)
        }
        if not self.single_head:
            result["head2_logits"] = self.policy_head2(policy_base)  # (B, 23)
            result["head3_logits"] = self.policy_head3(policy_base)  # (B, 23)
        return result

    def get_parameters_as_vector(self) -> np.ndarray:
        """Flatten all parameters into a single vector (for ES)."""
        params = []
        for p in self.parameters():
            params.append(p.data.view(-1).cpu().numpy())
        return np.concatenate(params)

    def set_parameters_from_vector(self, vec: np.ndarray) -> None:
        """Restore parameters from a flattened vector."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(torch.from_numpy(vec[idx: idx + size]).view(p.shape))
            idx += size

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(num_resblocks: int = 6, single_head: bool = False) -> AntWarNetwork:
    """Create a model and initialize all weights to very small random values."""
    model = AntWarNetwork(num_resblocks=num_resblocks, single_head=single_head)
    return model


def create_zero_model(num_resblocks: int = 6, single_head: bool = False) -> AntWarNetwork:
    """Create a model with all weights set to exactly zero (for testing)."""
    model = AntWarNetwork(num_resblocks=num_resblocks, single_head=single_head)
    for p in model.parameters():
        nn.init.zeros_(p)
    # Need to set some non-zero biases to avoid dead ReLU in testing
    # Actually, all-zero should work since we mask illegal actions.
    # Let's leave it all zero for the "pure zero" test.
    return model


def create_tuned_model() -> AntWarNetwork:
    """Create a model with hand-tuned weights to test specific behaviors.

    Strategy: set specific output channels to prefer certain actions.
    """
    model = create_zero_model()
    with torch.no_grad():
        # Make action_map[0] (build Basic) have higher values at position (4,2)
        # action_map shape: (23, 19, 19), but Conv weight shape is (23, 64, 1, 1)
        # We need to go through the whole network, so let's set the final biases instead.

        # Set the conv1×1 bias so channel 0 (build Basic) has a constant offset
        model.action_map_conv.bias[0] = 5.0  # bias for build Basic channel

        # Set head1_logits bias to prefer class 0
        model.policy_head1.bias[0] = 5.0

        # Set head2_logits bias to prefer class 16 (downgrade/demolish)
        model.policy_head2.bias[16] = 5.0

        # Set head3_logits bias to prefer class 17 (lightning storm)
        model.policy_head3.bias[17] = 5.0

    return model


if __name__ == "__main__":
    # Quick smoke test
    model = create_model()
    board = torch.randn(1, 28, 19, 19)
    stats = torch.randn(1, 22)
    out = model(board, stats)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"action_map:  {out['action_map'].shape}")
    print(f"head1_logits: {out['head1_logits'].shape}")
    print(f"head2_logits: {out['head2_logits'].shape}")
    print(f"head3_logits: {out['head3_logits'].shape}")
    print(f"value:        {out['value'].shape}")

    zero_model = create_zero_model()
    out_z = zero_model(board, stats)
    print(f"\nZero model output stats:")
    print(f"  action_map range: [{out_z['action_map'].min():.3f}, {out_z['action_map'].max():.3f}]")
    print(f"  value range:      [{out_z['value'].min():.3f}, {out_z['value'].max():.3f}]")

    tuned_model = create_tuned_model()
    out_t = tuned_model(board, stats)
    print(f"\nTuned model output stats:")
    print(f"  head1 argmax: {out_t['head1_logits'].argmax().item()} (expected 0)")
    print(f"  head2 argmax: {out_t['head2_logits'].argmax().item()} (expected 16)")
    print(f"  head3 argmax: {out_t['head3_logits'].argmax().item()} (expected 17)")
