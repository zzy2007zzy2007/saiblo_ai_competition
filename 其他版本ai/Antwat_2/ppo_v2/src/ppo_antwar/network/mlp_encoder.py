import torch
import torch.nn as nn

MLP_OUTPUT_DIM = 128


class MLPEncoder(nn.Module):
    """全局特征编码器 - 输入 GLOBAL_FEATURE_DIM 维，输出 MLP_OUTPUT_DIM 维"""

    GLOBAL_FEATURE_DIM = 33

    def __init__(
        self, input_dim: int = GLOBAL_FEATURE_DIM, hidden_dim: int = MLP_OUTPUT_DIM
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["MLPEncoder", "MLP_OUTPUT_DIM"]
