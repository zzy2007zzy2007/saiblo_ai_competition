from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .hex_conv import orthogonal_init
from .hex_cnn_encoder import HexCNNEncoder
from .mlp_encoder import MLPEncoder, MLP_OUTPUT_DIM
from .heads import ActionHead


class AntWarPolicyValueNetwork(nn.Module):
    """GA 策略网络 — 语义分段动作输出。

    本网络仅用于 GA 策略推断（子代生成/对战），不包含任何 PPO 组件。

    输入:
        board:      (batch, 29, 19, 19)  棋盘特征图
        global_vec: (batch, 33)           全局特征向量

    输出（通过 get_action）:
        strategy: int   ∈ {0,1,2}  战略选择 [防御/无操作/进攻]
        type_idx: int   ∈ {0..5}   类型选择
        sub_type: int   ∈ {0..9}   子类选择
        position: int   ∈ {0..7}   位置选择

    动作解码在外部（GAWarAgent/ActionMaskHandler）完成：
        (strategy, type_idx) → 查 ACTION_DECODE_TABLE → 动作名称
        position → 查多表位置映射 → 具体坐标
        (动作名称, 坐标) → construct_operation() → Operation
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # ── 策略编码器（唯一编码器） ──
        self.policy_cnn = HexCNNEncoder(hidden_dim)
        self.policy_mlp = MLPEncoder()
        cnn_out_channels = HexCNNEncoder.CHANNEL_COMPRESS  # 128
        cnn_out_dim = (
            cnn_out_channels * HexCNNEncoder.POOL_OUTPUT_SIZE * HexCNNEncoder.POOL_OUTPUT_SIZE
        )
        mlp_out_dim = MLP_OUTPUT_DIM
        fused_dim = cnn_out_dim + mlp_out_dim
        self.policy_proj = nn.Linear(fused_dim, hidden_dim)
        self.policy_layer_norm = nn.LayerNorm(hidden_dim)

        # ── 动作输出头 ──
        self.action_head = ActionHead(hidden_dim)

        self._init_weights()

    def _init_weights(self):
        orthogonal_init(self.policy_proj, gain=np.sqrt(2))

    def encode(self, board: torch.Tensor, global_vec: torch.Tensor) -> torch.Tensor:
        """单一编码器：棋盘 CNN + 全局 MLP → hidden_dim 向量

        Args:
            board:      (batch, 29, 19, 19)
            global_vec: (batch, 33)

        Returns:
            features:   (batch, hidden_dim)
        """
        cnn_out = self.policy_cnn(board)
        mlp_out = self.policy_mlp(global_vec)
        fused = torch.cat([cnn_out, mlp_out], dim=1)
        projected = self.policy_proj(fused)
        return self.policy_layer_norm(projected)

    @torch.no_grad()
    def get_action(
        self,
        board: torch.Tensor,
        global_vec: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """网络推理：输出 27 维向量并解码为四元组（确定性）。

        GA 无探索噪声需求，始终 argmax（详见 action_space_principle.md 第 5 条）。

        Args:
            board:      (1, 29, 19, 19) 或 (batch, ...)
            global_vec: (1, 33) 或 (batch, ...)

        Returns:
            (strategy, type_idx, sub_type, position) 四个整数
        """
        features = self.encode(board, global_vec)   # (1, hidden_dim)
        logits = self.action_head(features)         # (1, 27)

        strategy, type_idx, sub_type, position = self.action_head.decode(logits)

        return (
            strategy.item(),
            type_idx.item(),
            sub_type.item(),
            position.item(),
        )


def count_parameters(model: nn.Module) -> int:
    """统计模型可训练参数总数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["AntWarPolicyValueNetwork", "count_parameters"]
