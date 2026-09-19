from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hex_conv import orthogonal_init

# ── 27D 动作空间语义分段 ──
SEGMENTS = {
    "strategy": (0, 3),
    "type_idx": (3, 9),
    "sub_type": (9, 19),
    "position": (19, 27),
}


class ActionHead(nn.Module):
    """语义分段动作头 — 将 hidden_dim 维特征投影为 27 维向量。

    结构（2 个中间层 + ReLU + LayerNorm）:
        Linear(hidden_dim → hidden_dim) + LayerNorm + ReLU
        Linear(hidden_dim → hidden_dim // 2) + LayerNorm + ReLU
        Linear(hidden_dim // 2 → 27)

    27 维向量语义分段：
        [0:3)   strategy  logits  → argmax → {0,1,2}
        [3:9)   type      logits  → argmax → {0..5}
        [9:19)  sub_type  logits  → argmax → {0..9}
        [19:27) position  logits  → argmax → {0..7}

    注意：type_idx 无固有语义，需联合 strategy 查 ACTION_DECODE_TABLE 确定动作。
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 27),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m, gain=0.5)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播：输出 27 维原始 logits

        Args:
            features: (batch, hidden_dim) 编码器输出

        Returns:
            logits: (batch, 27) 未分段 softmax 的原始 logits
        """
        return self.net(features)

    @staticmethod
    def decode(logits: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """分段 argmax 解码（批处理版本）

        GA 确定性决策：softmax 不改变 argmax 结果，故直接对 logits 取 argmax。

        Args:
            logits: (batch, 27) 网络输出的原始 logits

        Returns:
            (strategy, type_idx, sub_type, position) 各 (batch,) 张量
        """
        strategy = torch.argmax(logits[:, 0:3], dim=-1)
        type_idx = torch.argmax(logits[:, 3:9], dim=-1)
        sub_type = torch.argmax(logits[:, 9:19], dim=-1)
        position = torch.argmax(logits[:, 19:27], dim=-1)
        return strategy, type_idx, sub_type, position


__all__ = [
    "ActionHead",
    "SEGMENTS",
]
