from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def orthogonal_init(layer: nn.Module, gain: float = 1.0) -> None:
    """正交初始化权重，偏置置零"""
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.LayerNorm):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)


class HexConv(nn.Module):
    """六边形卷积 - 使用六方向邻域 + einsum 实现"""

    HEX_DIRECTIONS: List[Tuple[int, int]] = [
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
    ]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, len(self.HEX_DIRECTIONS))
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.uniform_(self.bias, -0.1, 0.1)

    def get_hex_neighbor_sum(self, x: torch.Tensor) -> torch.Tensor:
        """对输入特征图沿六个方向提取邻域值并拼接"""
        B, C, H, W = x.shape
        neighbors = []
        for dr, dc in self.HEX_DIRECTIONS:
            shifted = torch.roll(x, shifts=(-dr, -dc), dims=(2, 3))
            if dr != 0:
                if dr > 0:
                    # 上移后环绕在底部
                    shifted[:, :, -dr:, :] = 0.0
                else:
                    # 下移后环绕在顶部
                    shifted[:, :, :-dr, :] = 0.0
            if dc != 0:
                if dc > 0:
                    # 左移后环绕在右侧
                    shifted[:, :, :, -dc:] = 0.0
                else:
                    # 右移后环绕在左侧
                    shifted[:, :, :, :-dc] = 0.0
            neighbors.append(shifted)
        return torch.stack(neighbors, dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """六边形卷积前向传播"""
        neighbor_sum = self.get_hex_neighbor_sum(x)
        out = torch.einsum("ocn,bcnhw->bohw", self.weight, neighbor_sum)
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


__all__ = ["HexConv", "orthogonal_init"]
