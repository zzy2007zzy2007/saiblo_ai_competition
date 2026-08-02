"""Neural network model for Ant-Game AI.

Architecture (AlphaZero-style):
  board(28,19,19) → Conv7×7 + ResBlock×6 → spatial_feat(64,19,19)
      ├── Conv1×1, 64→23 → action_map(23,19,19)
      └── GAP → board_emb(64)
  stats(~22) → MLP → stats_emb(64)
  state_emb = concat(board_emb, stats_emb) = 128
      ├── Value Head → scalar (bare linear)
      └── Policy Head → 3 × 23 class logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """Residual block with two Conv3×3 layers."""

    def __init__(self, channels: int, no_bn: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=no_bn)
        self.bn1 = nn.Identity() if no_bn else nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=no_bn)
        self.bn2 = nn.Identity() if no_bn else nn.BatchNorm2d(channels)

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

    ``num_heads`` controls how many policy heads are created (default 3).
    The forward method produces ``head1_logits, ..., headN_logits``.
    """

    NUM_CLASSES = 24  # 0-22 action classes + 23 = HOLD
    BOARD_CHANNELS = 28
    BOARD_SIZE = 19
    STATS_DIM = 42

    def __init__(self, num_resblocks: int = 6, num_heads: int = 3,
                 latent_dim: int = 64, no_bn: bool = False):
        super().__init__()
        self.LATENT_DIM = latent_dim
        self.num_resblocks = num_resblocks
        self.num_heads = num_heads
        self.no_bn = no_bn

        # Board encoder
        if no_bn:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(self.BOARD_CHANNELS, self.LATENT_DIM, kernel_size=7, padding=3, bias=True),
                nn.ReLU(),
            )
        else:
            self.initial_conv = nn.Sequential(
                nn.Conv2d(self.BOARD_CHANNELS, self.LATENT_DIM, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(self.LATENT_DIM),
                nn.ReLU(),
            )

        self.resblocks = nn.ModuleList([
            ResBlock(self.LATENT_DIM, no_bn=no_bn) for _ in range(num_resblocks)
        ])

        # Stats encoder
        self.stats_mlp = nn.Sequential(
            nn.Linear(self.STATS_DIM, self.LATENT_DIM),
            nn.ReLU(),
        )

        # Board embedding is passed through as-is (no LayerNorm).
        # LayerNorm was removed because it destroyed inter-sample variance
        # (encoder output variance is primarily in magnitude, not direction).

        # Policy head - spatial
        self.action_map_conv = nn.Conv2d(self.LATENT_DIM, self.NUM_CLASSES, kernel_size=1)

        # Policy head - class
        self.policy_dropout = nn.Dropout(p=0.5)
        self.policy_base = nn.Sequential(
            nn.Linear(self.LATENT_DIM, self.LATENT_DIM),
            nn.ReLU(),
        )
        self.policy_heads = nn.ModuleList([
            nn.Linear(self.LATENT_DIM, self.NUM_CLASSES) for _ in range(num_heads)
        ])

        # Value head — bare linear output (no Tanh).  The value target
        # (discounted sum of HP differences) can exceed [-1, 1], so Tanh
        # would cap the value network's range and inflate GAE error.
        self.value_head = nn.Sequential(
            nn.Linear(self.LATENT_DIM * 2, self.LATENT_DIM),  # state_emb = board_emb(64) + stats_emb(64)
            nn.ReLU(),
            nn.Linear(self.LATENT_DIM, 1),
        )

    def forward(self, board: torch.Tensor, stats: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            board: (B, 28, 19, 19)
            stats: (B, 22)
        Returns:
            dict with keys:
              - action_map: (B, 23, 19, 19)
              - state_emb: (B, 128) — board_emb + stats_emb
              - head1_logits, ..., headN_logits: (B, 23) — one per head, N = self.num_heads
              - value: (B, 1)
        """
        # Board encoder
        x = self.initial_conv(board)  # (B, 64, 19, 19)
        for block in self.resblocks:
            x = block(x)  # (B, 64, 19, 19)
        spatial_feat = x  # keep for action map

        # Global average pooling → board embedding
        board_emb = spatial_feat.mean(dim=[2, 3])  # (B, 64)

        # No LayerNorm — board_emb retains its natural variance

        # Stats encoder
        stats_emb = self.stats_mlp(stats)  # (B, 64)

        # State embedding
        state_emb = torch.cat([board_emb, stats_emb], dim=1)  # (B, 128)

        # --- Policy head ---
        # Spatial: action map (shared)
        action_map = self.action_map_conv(spatial_feat)  # (B, 23, 19, 19)

        # Class heads (with per-head dropout to prevent shortcut learning)
        policy_base = self.policy_base(board_emb)  # (B, 64)
        result: dict[str, torch.Tensor] = {
            "action_map": action_map,
            "state_emb": state_emb,  # (B, 128) — used by gating network
            "value": self.value_head(state_emb),  # (B, 1)
        }
        for i in range(self.num_heads):
            # Per-head dropout: each head sees a different dropout mask
            # This prevents all heads from converging to the same shortcut
            head_input = self.policy_dropout(policy_base)  # (B, 64)
            result[f"head{i+1}_logits"] = self.policy_heads[i](head_input)  # (B, 23)
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

    @staticmethod
    def fold_bn_into_state_dict(sd: dict[str, torch.Tensor], eps: float = 1e-5) -> dict[str, torch.Tensor]:
        """Fold BatchNorm parameters into preceding Conv2d layers.

        Returns a state_dict compatible with ``no_bn=True`` models.
        BN weight/bias/running_mean/running_var keys are removed.
        Conv bias keys are added where BN existed.
        """
        import re, copy
        sd = copy.copy(sd)

        # Pattern: for each Conv2d that has a matching BN immediately after
        conv_bn_pairs = [
            ("initial_conv.0", "initial_conv.1"),
        ]
        for i in range(6):
            conv_bn_pairs.append((f"resblocks.{i}.conv1", f"resblocks.{i}.bn1"))
            conv_bn_pairs.append((f"resblocks.{i}.conv2", f"resblocks.{i}.bn2"))

        for conv_prefix, bn_prefix in conv_bn_pairs:
            w_key = f"{conv_prefix}.weight"
            if w_key not in sd:
                continue  # no_bn already, skip

            w = sd[w_key]  # Conv weight
            gamma = sd[f"{bn_prefix}.weight"]  # BN gamma
            beta = sd[f"{bn_prefix}.bias"]      # BN beta
            rm = sd[f"{bn_prefix}.running_mean"]
            rv = sd[f"{bn_prefix}.running_var"]

            # Fold: W' = gamma * W / sqrt(var + eps)
            scale = gamma / torch.sqrt(rv + eps)
            folded_w = w * scale.reshape(-1, 1, 1, 1)
            sd[w_key] = folded_w

            # Fold: b' = beta - gamma * mean / sqrt(var + eps)
            folded_b = beta - scale * rm
            sd[f"{conv_prefix}.bias"] = folded_b

            # Remove BN keys
            for k in list(sd.keys()):
                if k.startswith(bn_prefix):
                    del sd[k]

        return sd

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_model(num_resblocks: int = 6, num_heads: int = 3,
                 latent_dim: int = 64, small: bool = False,
                 no_bn: bool = False) -> AntWarNetwork:
    """Create a model.

    Args:
        num_resblocks: Number of residual blocks.
        num_heads: Number of policy heads.
        latent_dim: Latent dimension (ignored if small=True).
        small: If True, creates a smaller model (latent_dim=32,
               num_resblocks=2, 1 head) suitable for limited data.
    """
    if small:
        latent_dim = 32
        num_resblocks = 2
        num_heads = 1
    model = AntWarNetwork(num_resblocks=num_resblocks, num_heads=num_heads,
                          latent_dim=latent_dim, no_bn=no_bn)
    return model


def create_zero_model(num_resblocks: int = 6, num_heads: int = 3) -> AntWarNetwork:
    """Create a model with all weights set to exactly zero (for testing)."""
    model = AntWarNetwork(num_resblocks=num_resblocks, num_heads=num_heads)
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
        model.policy_heads[0].bias[0] = 5.0

        # Set head2_logits bias to prefer class 16 (downgrade/demolish)
        model.policy_heads[1].bias[16] = 5.0

        # Set head3_logits bias to prefer class 17 (lightning storm)
        model.policy_heads[2].bias[17] = 5.0

    return model


if __name__ == "__main__":
    # Quick smoke test
    model = create_model()
    board = torch.randn(1, 28, 19, 19)
    stats = torch.randn(1, 22)
    out = model(board, stats)
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"action_map:  {out['action_map'].shape}")
    print(f"state_emb:   {out['state_emb'].shape}")
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
