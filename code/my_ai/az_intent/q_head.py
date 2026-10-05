"""类条件价值头 **Q(s, c)**（方法侧 M2，预注册 `docs/prereg_20261005_qhead_class_decision.md`）。

动机（实测）：行为克隆（argmax 预测概率）**结构上与「这么打赢不赢」无关** —— 修好决策机制
（`pos_pin=masked`）、修好标签（`--label executed`）、加反频率权重都试过：
`power=0.5` ⇒ 不花钱（行为=A1、`p̂=0.5000`）；`power=1.0` ⇒ 过度花钱（建塔 26.3、闪电 8.33、`p̂=0.3750`）。
⇒ 换**目标函数**：直接回归「在这个局面上出这个类，最后赢不赢」。

结构（刻意做得极小、便于验证）：
    `emb = 冻结主干 class 网的 state_emb (128)`  ⊕  `类 one-hot (24)`
    →  Linear(152, 64) → ReLU → Linear(64, 1) → **Q(s, c)**（对局结果的预测，∈[-1,1]）

只在 `class_state` 主干**冻结**的前提下训练（主干逐位不动）⇒ 训练与复现都便宜。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

EMB_DIM = 128
N_CLS = 24


class QHead(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM, n_cls: int = N_CLS, hidden: int = 64):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.n_cls = int(n_cls)
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(self.emb_dim + self.n_cls, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self, emb: torch.Tensor, cls_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, cls_onehot], dim=-1)
        return self.net(x).squeeze(-1)

    def score_all(self, emb: torch.Tensor) -> torch.Tensor:
        """(B, emb) -> (B, n_cls)：对 24 个类各打一个分（内部构造 one-hot）。"""
        eye = torch.eye(self.n_cls, device=emb.device, dtype=emb.dtype)
        e = emb.unsqueeze(1).expand(-1, self.n_cls, -1)
        o = eye.unsqueeze(0).expand(emb.shape[0], -1, -1)
        return self.forward(e.reshape(-1, self.emb_dim), o.reshape(-1, self.n_cls)
                            ).reshape(emb.shape[0], self.n_cls)

    @torch.no_grad()
    def score_np(self, emb: np.ndarray) -> np.ndarray:
        """(emb_dim,) -> (n_cls,) —— 给部署路径用（numpy，单个局面）。"""
        self.eval()
        e = torch.from_numpy(np.asarray(emb, dtype=np.float32)).unsqueeze(0)
        return self.score_all(e).squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "emb_dim": self.emb_dim,
                    "n_cls": self.n_cls, "hidden": self.hidden}, path)

    @staticmethod
    def load(path: str) -> "QHead":
        d = torch.load(path, map_location="cpu", weights_only=False)
        m = QHead(emb_dim=int(d["emb_dim"]), n_cls=int(d["n_cls"]), hidden=int(d["hidden"]))
        m.load_state_dict(d["state_dict"])
        m.eval()
        return m
