"""实体 token Transformer —— 用于价值函数（`docs/plan_entity_transformer_value.md` 第 1 步）。

设计（token 化，单位是**实体**而不是格子）：
  - **塔 token**（每塔一个）：`[is_own, hp/max_hp, cooldown]` + `Emb(tower_type)` + `Pos2D(x,y)` + `TypeEmb(TOWER)`
  - **蚁 token**（每蚁一个）：`[is_own, hp/max_hp, level/2, age/64]` + `Emb(kind)` + `Emb(behavior)`
                             + `Pos2D(x,y)` + `TypeEmb(ANT)`
  - **stats token**（1 个）：`Linear(42)` + `TypeEmb(STATS)`
  - **CLS token**（1 个）：可学习向量 + `TypeEmb(CLS)`
价值 = CLS token 过 MLP。

为什么这么设计：`is_own` 是相对"该玩家视角"的；塔的 type/hp 与蚁的 kind/hp 是现有 28 通道 board 里
**没有**的（board 只有 level/射程/伤害；蚁按格求和、缺 kind），实体 token 把它们原样带上 ⇒ 无损。
位置用**学习式二维位置 embedding**（固定 19×19，比 RoPE 简单）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

# TowerType 的离散值不连续，列出来映射成 0..16
TOWER_TYPES = [0, 1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43]
TOWER_TYPE_TO_ID = {t: i for i, t in enumerate(TOWER_TYPES)}
N_TOWER_TYPES = len(TOWER_TYPES) + 1        # +1 = pad
N_ANT_KINDS = 3                             # worker/combat + pad
N_ANT_BEHAVIORS = 6                         # 0-4 + pad

TOWER_CONT_DIM = 3    # is_own, hp_ratio, cooldown
ANT_CONT_DIM = 4      # is_own, hp_ratio, level, age
PAD_CELL = 361        # position embedding 的 pad 下标（19*19=361 个格）


class EntityTransformerValue(nn.Module):
    def __init__(self, d_model: int = 128, n_layers: int = 3, n_heads: int = 4,
                 ffn: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.tower_cont = nn.Linear(TOWER_CONT_DIM, d_model)
        self.tower_type = nn.Embedding(N_TOWER_TYPES, d_model)
        self.ant_cont = nn.Linear(ANT_CONT_DIM, d_model)
        self.ant_kind = nn.Embedding(N_ANT_KINDS, d_model)
        self.ant_behavior = nn.Embedding(N_ANT_BEHAVIORS, d_model)
        self.pos = nn.Embedding(19 * 19 + 1, d_model)   # +1 pad
        self.type_emb = nn.Embedding(4, d_model)        # 0=CLS 1=STATS 2=TOWER 3=ANT
        self.stats_proj = nn.Linear(42, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """batch keys: tower_cont(B,Tn,3), tower_type(B,Tn) long, tower_pos(B,Tn) long, tower_mask(B,Tn)
        ant_cont(B,An,4), ant_kind(B,An), ant_behavior(B,An), ant_pos(B,An), ant_mask(B,An)
        stats(B,42).  返回 (B,1)。"""
        B = batch["stats"].shape[0]
        dev = batch["stats"].device
        toks, masks = [], []

        # CLS + stats
        cls = self.cls.expand(B, 1, -1) + self.type_emb.weight[0]
        toks.append(cls); masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))
        st = self.stats_proj(batch["stats"]).unsqueeze(1) + self.type_emb.weight[1]
        toks.append(st); masks.append(torch.ones(B, 1, dtype=torch.bool, device=dev))

        # towers
        tw = (self.tower_cont(batch["tower_cont"])
              + self.tower_type(batch["tower_type"])
              + self.pos(batch["tower_pos"])
              + self.type_emb.weight[2])
        toks.append(tw); masks.append(batch["tower_mask"])
        # ants
        an = (self.ant_cont(batch["ant_cont"])
              + self.ant_kind(batch["ant_kind"])
              + self.ant_behavior(batch["ant_behavior"])
              + self.pos(batch["ant_pos"])
              + self.type_emb.weight[3])
        toks.append(an); masks.append(batch["ant_mask"])

        x = torch.cat(toks, dim=1)                  # (B, N, d)
        keep = torch.cat(masks, dim=1)              # (B, N) bool
        # transformer 的 key_padding_mask: True = 忽略
        out = self.encoder(x, src_key_padding_mask=~keep)
        out = self.norm(out)
        cls_out = out[:, 0]
        return self.head(cls_out)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
