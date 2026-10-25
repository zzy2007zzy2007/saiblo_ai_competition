"""构造 hybrid checkpoint：prior_z_80g.best 全套，**只把闪电通道(17)换成 f10 的**。

用法：python make_hybrid_prior.py  → checkpoints/prior_z_hybrid_light.pt
原理：`action_map_conv` 的输出层有 24 个通道（weight 的 out_channels 维、bias 的第 0 维）；
      把第 17 行（闪电）替换为 f10 的对应行，其余原样保留。
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import torch

NEW = REPO / "checkpoints/prior_z_80g.best.pt"
OLD = REPO / "checkpoints/prior_distill_f10.pt"
OUT = REPO / "checkpoints/prior_z_hybrid_light.pt"
CH = 17  # 闪电


def main() -> int:
    a = torch.load(NEW, map_location="cpu", weights_only=False)
    b = torch.load(OLD, map_location="cpu", weights_only=False)
    sa, sb = a["pos_state"], b["pos_state"]
    spliced = []
    for k, v in sa.items():
        if not isinstance(v, torch.Tensor) or v.ndim < 1 or v.shape[0] != 24:
            continue
        if k not in sb or sb[k].shape != v.shape:
            print(f"  ⚠️ 跳过 {k}（形状不一致或缺失）")
            continue
        nv = v.clone()
        nv[CH] = sb[k][CH]
        sa[k] = nv
        spliced.append(f"{k}{tuple(v.shape)}")
    a["pos_state"] = sa
    a["note"] = "hybrid: prior_z_80g.best + f10 的闪电通道(17)"
    torch.save(a, OUT)
    print(f"  已替换 {len(spliced)} 个张量的通道 {CH}:")
    for s in spliced:
        print(f"    {s}")
    print(f"  写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
