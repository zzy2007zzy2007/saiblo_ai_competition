import numpy as np
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .hex_conv import HexConv, orthogonal_init


class HexCNNEncoder(nn.Module):
    """六边形卷积编码器。

    输入 (BOARD_CHANNELS, BOARD_SIZE, BOARD_SIZE)，经三阶段六边形卷积 +
    通道压缩后输出展平特征向量。

    输出维度 = CHANNEL_COMPRESS × POOL_OUTPUT_SIZE × POOL_OUTPUT_SIZE
    """

    BOARD_CHANNELS = 28
    BOARD_SIZE = 19
    POOL_OUTPUT_SIZE = 5
    CHANNEL_COMPRESS = 128       # 压缩后通道数

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        c1, c2, c3 = hidden_dim // 2, hidden_dim, hidden_dim * 2

        # ── 阶段1：初始投影 + 六边形卷积 ──
        self.stage1 = nn.Sequential(
            nn.Conv2d(self.BOARD_CHANNELS, c1, kernel_size=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.hex_conv1 = HexConv(c1, c1)

        # ── 阶段2：下采样 + 六边形卷积 ──
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )
        self.hex_conv2 = HexConv(c2, c2)

        # ── 阶段3：下采样 + 六边形卷积 ──
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )
        self.hex_conv3 = HexConv(c3, c3)

        # ── 通道压缩层（1×1 Conv）──
        # 将 512 通道压缩至 128 通道，减少后续融合层参数量
        self.channel_compress = nn.Sequential(
            nn.Conv2d(c3, self.CHANNEL_COMPRESS, kernel_size=1),
            nn.BatchNorm2d(self.CHANNEL_COMPRESS),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(
            (self.POOL_OUTPUT_SIZE, self.POOL_OUTPUT_SIZE)
        )

        self._init_weights()

    def _init_weights(self):
        """初始化新增层的权重。

        初始化策略：

        1. Conv2d（1×1 通道压缩）：使用正交初始化（gain=√2），与项目中
           Fusion Projection 层的 Linear 保持一致。正交初始化确保权重矩阵的
           行（输出通道方向）近似正交，使得：
           - 各输出通道提取互补而非冗余的特征
           - 前向传播时特征方差稳定，反向传播时梯度方差不衰减
           - gain=√2 适配 ReLU 激活函数，补偿其将负半轴置零带来的方差减半

        2. BatchNorm2d：使用 PyTorch 默认初始化（weight=1, bias=0），
           不做额外处理。BatchNorm 在前几个 batch 的统计量稳定后会自动
           调节缩放和偏移，因此初始化值不影响最终收敛结果。

        3. 现有层（stage1/2/3、hex_conv1/2/3）已在各自构造器中完成初始化，
           此处不重复处理。
        """
        orthogonal_init(self.channel_compress[0], gain=np.sqrt(2))
        # BatchNorm2d 使用默认初始化 (weight=1, bias=0)，无需显式调用

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = x + self.hex_conv1(x)

        x = self.stage2(x)
        x = x + self.hex_conv2(x)

        x = self.stage3(x)
        x = x + self.hex_conv3(x)

        x = self.channel_compress(x)   # 512 通道 → 128 通道

        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return x


__all__ = ["HexCNNEncoder"]
