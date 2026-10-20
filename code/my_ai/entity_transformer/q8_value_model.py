"""带「计划 token」的实体 Transformer —— 预测 `mean_terminal_score`（8 回合期望值）。

和 `model.py:EntityTransformerValue` 的差别：
- `stats` 维度可配置（这里用 63 维聚合状态特征）；
- 多一个 **plan token**：`Linear(41) + TypeEmb(PLAN)`，把候选计划（建/升级/降级在哪个格、
  升级目标塔型、是否 hold…）编码进去；
- 输出仍然是 CLS 的标量。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from my_ai.entity_transformer.model import (
    ANT_CONT_DIM, PAD_CELL, TOWER_CONT_DIM, N_ANT_BEHAVIORS, N_ANT_KINDS, N_TOWER_TYPES,
)


class PlanConditionedEntityValue(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 ffn: int = 256, dropout: float = 0.1,
                 state_dim: int = 63, plan_dim: int = 41, pos_mode: str = "learned"):
        super().__init__()
        assert pos_mode in ("learned", "xy", "none"), pos_mode
        self.d_model = d_model
        self.pos_mode = pos_mode
        self.tower_cont = nn.Linear(TOWER_CONT_DIM, d_model)
        self.tower_type = nn.Embedding(N_TOWER_TYPES, d_model)
        self.ant_cont = nn.Linear(ANT_CONT_DIM, d_model)
        self.ant_kind = nn.Embedding(N_ANT_KINDS, d_model)
        self.ant_behavior = nn.Embedding(N_ANT_BEHAVIORS, d_model)
        if pos_mode == "learned":
            self.pos = nn.Embedding(19 * 19 + 1, d_model)
        elif pos_mode == "xy":
            self.pos_x = nn.Embedding(19 + 1, d_model)
            self.pos_y = nn.Embedding(19 + 1, d_model)
        # 0=CLS 1=STATE 2=TOWER 3=ANT 4=PLAN
        self.type_emb = nn.Embedding(5, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.plan_proj = nn.Linear(plan_dim, d_model)
        self.plan_pos = nn.Embedding(19 * 19 + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))

    def _pos_embed(self, pos_ids: torch.Tensor) -> torch.Tensor:
        if self.pos_mode == "learned":
            return self.pos(pos_ids)
        if self.pos_mode == "xy":
            px = (pos_ids // 19).clamp(max=19)
            py = (pos_ids % 19).clamp(max=19)
            return self.pos_x(px) + self.pos_y(py)
        return 0.0

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch["state_feats"].shape[0]
        device = batch["state_feats"].device
        toks, masks = [], []

        cls = self.cls.expand(batch_size, 1, -1) + self.type_emb.weight[0]
        toks.append(cls)
        masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        state = self.state_proj(batch["state_feats"]).unsqueeze(1) + self.type_emb.weight[1]
        toks.append(state)
        masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        plan = (self.plan_proj(batch["plan_feats"])
                + self.plan_pos(batch["plan_pos"])
                + self.type_emb.weight[4]).unsqueeze(1)
        toks.append(plan)
        masks.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))

        tower = (self.tower_cont(batch["tower_cont"])
                 + self.tower_type(batch["tower_type"])
                 + self._pos_embed(batch["tower_pos"])
                 + self.type_emb.weight[2])
        toks.append(tower)
        masks.append(batch["tower_mask"])

        ant = (self.ant_cont(batch["ant_cont"])
               + self.ant_kind(batch["ant_kind"])
               + self.ant_behavior(batch["ant_behavior"])
               + self._pos_embed(batch["ant_pos"])
               + self.type_emb.weight[3])
        toks.append(ant)
        masks.append(batch["ant_mask"])

        x = torch.cat(toks, dim=1)
        keep = torch.cat(masks, dim=1)
        out = self.encoder(x, src_key_padding_mask=~keep)
        out = self.norm(out)
        return self.head(out[:, 0])


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
