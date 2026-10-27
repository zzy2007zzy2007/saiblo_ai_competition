"""`AZAI_VALUE_CKPT`：把 rollout 的**叶子值网**换成独立 `ValueNetwork`（默认关）。

## 背景（`docs/small_value_net_20261027.md` §5 工程层）

小值网 `ValueNetwork(32,3)`（103,137 参数）离线过了关（留出 R² 0.4716、与现役值网
`Spearman ρ(tanh)=0.9603`、前向 5.1×）。但它**塞不进三网 ckpt**（`load_three_models` 用同一组
`latent_dim/num_resblocks` 建三个网），而且两处消费点写死 `model(...)["value"]`
（`value_batch.py:55`、`flat_rollout_greedy.py:284`），`ValueNetwork` 却返回**裸张量**。
所以需要两样东西：①一个取值函数兼容两种形态；②一个开关，把"叶子值网"这个对象整体换掉。

## 语义

* `AZAI_VALUE_CKPT` **为空**（默认）⇒ `maybe_override(model)` **原样返回传入的同一个对象**，
  不 import、不加载任何模型 ⇒ 消费点行为**逐位不变**（证据：`code/test_match/
  probe_oracle_inject_zero_impact.py` 的哈希日志改动前/后逐行相同）。
* `AZAI_VALUE_CKPT=<path>` ⇒ 按该 ckpt 的 `config` 建 `ValueNetwork(width, blocks)`，
  若 ckpt 没标记 `y_folded` 就把 `y_std/y_mean` 折进 `value_head.2`（见下），
  再包一层 `ValueModelAdapter`：`__call__(board, stats)` 返回 `{"value": tensor}`
  （与 `AntWarNetwork` 同形态），`parameters()/eval()/to()/state_dict()` 透传
  ⇒ 现有 `next(model.parameters()).is_cuda`、`model.eval()` 都照旧工作。

⚠️ **只换叶子值网**：`net_fn`（三网，M6c 仲裁里那个 `net_out["value"]`）不动。
⚠️ 相对路径按**仓库根**解析（bridge 用 `cwd=code/test_match` 起进程，与 `AZAI_CKPT` 同一个坑）。

## 为什么要在这里再折一次

部署侧 `AZAI_VALUE_TANH=1` 只做 `tanh`、**不做反归一化**；而 `ValueNetwork` 是在
z-score 标签 `(y-y_mean)/y_std` 上训的（本次 `y_std=0.3861`）。不折就把输出放大 ≈2.6×，
`|v|>0.5` 的叶子会被 tanh 压平（= `docs/value_net_pipeline_survey.md` §5.3 的历史事故）。
`fold_small_value_ckpt.py` 产出的 `v_32_3_folded.pt` 已经折过并标记 `y_folded=True`；
这里只在**没折过**的 ckpt 上补折，避免折两次（折两次 = 量纲又被缩 2.6×）。
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "AZAI_VALUE_CKPT"

_REPO = Path(__file__).resolve().parents[3]   # code/my_ai/az_intent/ -> 仓库根
_CACHE: dict[str, object] = {}


def env_path() -> str:
    """当前 `AZAI_VALUE_CKPT` 的值（空串 = 关）。"""
    return str(os.environ.get(ENV_VAR, "") or "")


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (_REPO / p)


class ValueModelAdapter:
    """把独立 `ValueNetwork`（裸张量输出）包成三网形态（`{"value": tensor}`）。

    只做"形态适配 + 转发"，不改数值。`is_value_override=True` 供日志/自检识别。
    """

    is_value_override = True

    def __init__(self, net, *, path: str, y_std: float, y_mean: float, folded_here: bool):
        self.net = net
        self.path = path
        self.y_std = float(y_std)
        self.y_mean = float(y_mean)
        self.folded_here = bool(folded_here)

    # ── 数值接口 ─────────────────────────────────────────────────────────────
    def __call__(self, board, stats):
        return {"value": self.net(board, stats)}

    # ── nn.Module 接口透传（`next(model.parameters()).is_cuda` 依赖它）──────────
    def parameters(self, *a, **kw):
        return self.net.parameters(*a, **kw)

    def named_parameters(self, *a, **kw):
        return self.net.named_parameters(*a, **kw)

    def state_dict(self, *a, **kw):
        return self.net.state_dict(*a, **kw)

    def load_state_dict(self, *a, **kw):
        return self.net.load_state_dict(*a, **kw)

    def eval(self):
        self.net.eval()
        return self

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def to(self, *a, **kw):
        self.net.to(*a, **kw)
        return self

    def cpu(self):
        self.net.cpu()
        return self

    def __getattr__(self, name):        # 其余属性一律转发（如 num_heads / LATENT_DIM）
        return getattr(self.net, name)

    def describe(self) -> str:
        _p = sum(p.numel() for p in self.net.parameters())
        return (f"[value_override] 叶子值网 = {Path(self.path).name} "
                f"ValueNetwork(params={_p:,}) y_std={self.y_std:.4f} y_mean={self.y_mean:.4f} "
                f"{'（本进程内补折）' if self.folded_here else '（ckpt 已折）'}")


def value_of(out):
    """取值兼容：三网返回 `{"value": ...}`，独立 `ValueNetwork` 返回**裸张量**。

    dict 分支返回的就是原来那个张量对象 ⇒ 默认路径**逐位不变**。
    """
    if isinstance(out, dict):
        return out["value"]
    return out


def _fold_state_dict(sd: dict, y_std: float, y_mean: float) -> dict:
    """`out_new = out_old * y_std + y_mean` 折进 `value_head.2`（照 `graft_value_ckpt.py:68-76`）。"""
    out = {k: v.clone() for k, v in sd.items()}
    wk, bk = "value_head.2.weight", "value_head.2.bias"
    if wk not in out or bk not in out:
        raise KeyError(f"AZAI_VALUE_CKPT：ckpt 缺少 {wk}/{bk}，无法折归一化")
    out[wk] = out[wk] * float(y_std)
    out[bk] = out[bk] * float(y_std) + float(y_mean)
    return out


def load_override(path: str | None = None):
    """按 `AZAI_VALUE_CKPT`（或显式 `path`）加载独立值网并包成三网形态。**同路径只加载一次**。"""
    import torch

    from my_ai.value_network import ValueNetwork

    raw = (path or env_path()).strip()
    if not raw:
        raise RuntimeError("load_override: AZAI_VALUE_CKPT 为空（调用方应先判断）")
    p = _abs(raw)
    key = str(p.resolve())
    if key in _CACHE:
        return _CACHE[key]
    if not p.exists():
        raise FileNotFoundError(f"AZAI_VALUE_CKPT 指向的文件不存在: {p}")

    ck = torch.load(p, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or {}
    net = ValueNetwork(latent_dim=int(cfg.get("latent_dim", 64)),
                       num_resblocks=int(cfg.get("num_resblocks", 6)))
    sd = ck["state_dict"]
    y_std = float(ck.get("y_std", 1.0))
    y_mean = float(ck.get("y_mean", 0.0))
    folded_here = not bool(ck.get("y_folded", False)) and (y_std != 1.0 or y_mean != 0.0)
    if folded_here:
        sd = _fold_state_dict(sd, y_std, y_mean)
    net.load_state_dict(sd)
    net.eval()
    adapter = ValueModelAdapter(net, path=key, y_std=y_std, y_mean=y_mean,
                               folded_here=folded_here)
    _CACHE[key] = adapter
    return adapter


def maybe_override(model, path: str | None = None):
    """`AZAI_VALUE_CKPT` 为空 ⇒ **返回同一个对象**（默认行为逐位不变）；非空 ⇒ 返回适配器。"""
    raw = (path if path is not None else env_path()).strip()
    if not raw:
        return model
    return load_override(raw)


def describe(model) -> str:
    """给日志用：是覆盖网就返回一行说明，否则空串。"""
    if getattr(model, "is_value_override", False):
        return model.describe()
    return ""
