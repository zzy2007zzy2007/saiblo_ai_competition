import torch
import torch.nn as nn

from ..utils.action_constants import TYPE_CONFIG, NUM_HORIZONS
from .hex_conv import orthogonal_init

AUX_HEAD_HIDDEN = 64


class StructuredActionHead(nn.Module):
    """结构化动作头 - 将 119 维动作空间分层为类型 + 目标。

    对外接口:
        calc_type_logits(features)         → type_logits (batch, 9)
        calc_all_logits(features)          → (type_logits, target_logits_list)
        calc_target_logits(features, tid)  → target_logits (batch, n_valid_tid)
        mask_target_logits(t_logits, mask, tid) → target_logits (in-place)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        num_types = len(TYPE_CONFIG)
        self.type_head = nn.Linear(hidden_dim, num_types)

        self.target_heads = nn.ModuleList()
        for type_cfg in TYPE_CONFIG:
            self.target_heads.append(nn.Linear(hidden_dim, type_cfg["max_targets"]))

        self._init_weights()

    def _init_weights(self):
        orthogonal_init(self.type_head, gain=0.5)
        for _, head in enumerate(self.target_heads):
            gain = 0.5 if head.out_features >= 2 else 0.01
            orthogonal_init(head, gain=gain)

    # ── 对外接口 ──────────────────────────────────────────────────────────

    def calc_type_logits(self, features: torch.Tensor) -> torch.Tensor:
        """仅计算动作类型 logits。"""
        return self.type_head(features)  # (batch, 9)

    def calc_all_logits(
        self, features: torch.Tensor
    ):
        """计算 type logits + 全部 9 个类型的 target logits。

        Returns:
            type_logits: (batch, 9)
            target_logits_list: List[(batch, n_i)]，长度 9，每个已截断到有效维度
        """
        type_logits = self.type_head(features)
        target_list = [
            self.calc_target_logits(features, tid)
            for tid in range(len(TYPE_CONFIG))
        ]
        return type_logits, target_list

    def calc_target_logits(
        self, features: torch.Tensor, type_id: int
    ) -> torch.Tensor:
        """计算单个动作类型的 target logits（已截断到有效维度）。

        Args:
            features: (batch, hidden_dim)
            type_id: 动作类型索引 (0-8)

        Returns:
            target_logits: (batch, n_valid)，已截断
        """
        cfg = TYPE_CONFIG[type_id]
        n_valid = cfg["flat_end"] - cfg["flat_start"]
        t = self.target_heads[type_id](features)
        return t[:, :n_valid]

    def mask_target_logits(
        self,
        target_logits: torch.Tensor,
        action_mask: torch.Tensor,
        type_id: int,
    ) -> torch.Tensor:
        """原地对 target logits 施加 action_mask。

        Args:
            target_logits: (batch, n_valid)，calc_target_logits() 的返回值
            action_mask: (batch, 119)
            type_id: 动作类型索引

        Returns:
            原地修改后的 target_logits（同时返回引用）
        """
        cfg = TYPE_CONFIG[type_id]
        n_valid = target_logits.size(1)
        tm = action_mask[:, cfg["flat_start"] : cfg["flat_start"] + n_valid]
        target_logits.masked_fill_(tm == 0, float("-inf"))
        return target_logits

    # ── 静态工具方法 ─────────────────────────────────────────────────────

    @staticmethod
    def flat_mask_to_type_mask(flat_mask: torch.Tensor) -> torch.Tensor:
        batch_size = flat_mask.size(0)
        device = flat_mask.device
        num_types = len(TYPE_CONFIG)
        type_mask = torch.zeros(
            batch_size, num_types, device=device, dtype=flat_mask.dtype
        )
        for type_id, cfg in enumerate(TYPE_CONFIG):
            segment = flat_mask[:, cfg["flat_start"] : cfg["flat_end"]]
            type_mask[:, type_id] = (segment > 0.5).any(dim=1).to(flat_mask.dtype)
        return type_mask

    @staticmethod
    def flat_mask_to_target_mask(flat_mask: torch.Tensor, type_id: int) -> torch.Tensor:
        cfg = TYPE_CONFIG[type_id]
        n_valid = cfg["flat_end"] - cfg["flat_start"]
        return flat_mask[:, cfg["flat_start"] : cfg["flat_start"] + n_valid]


class ValueHead(nn.Module):
    """状态价值头"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)
        orthogonal_init(self.linear, gain=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TowerDamageHead(nn.Module):
    """塔伤害预测辅助头 - 预测多个时间视界的塔伤害"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, AUX_HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(AUX_HEAD_HIDDEN, NUM_HORIZONS * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoldIncomeHead(nn.Module):
    """金币收入预测辅助头 - 预测多个时间视界的金币收入"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, AUX_HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(AUX_HEAD_HIDDEN, NUM_HORIZONS * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseDamageHead(nn.Module):
    """基地伤害预测辅助头 - 预测多个时间视界的基地 HP 伤害"""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, AUX_HEAD_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Linear(AUX_HEAD_HIDDEN, NUM_HORIZONS * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


__all__ = ["StructuredActionHead", "ValueHead", "TowerDamageHead", "GoldIncomeHead", "BaseDamageHead"]
