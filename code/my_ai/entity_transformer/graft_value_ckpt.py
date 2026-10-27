"""把 `ValueNetwork`（低质量大数据训的价值网）的权重灌进 M6c ckpt 的 `value_state`。

两者层名完全一致（initial_conv / resblocks / stats_mlp / value_head），
`value_state` 是完整的 AntWarNetwork state_dict（94 项），我们只覆盖 value 通路的键，
policy 头保持原样（用 strict=False）。

用法：
  python code/my_ai/entity_transformer/graft_value_ckpt.py \
      --ckpt checkpoints/posnet_A_k5_m32.pt \
      --value checkpoints/lowq_value_seed0.pt \
      --out checkpoints/posnet_A_k5_m32_lowq.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "code"))
sys.path.insert(0, str(_REPO / "Ant-Game"))

from my_ai.value_network import ValueNetwork  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = Path(a.ckpt)
    if not base.is_absolute():
        base = _REPO / base
    vp = Path(a.value)
    if not vp.is_absolute():
        vp = _REPO / vp
    out = Path(a.out)
    if not out.is_absolute():
        out = _REPO / out

    ck = torch.load(base, map_location="cpu", weights_only=False)
    vk = torch.load(vp, map_location="cpu", weights_only=False)
    # ⚠️ 按被嫁网自己的 `config` 建网（否则把 32/3 装进 64/6 只会得到"shape mismatch"警告）
    _cfg = vk.get("config") or {}
    vnet = ValueNetwork(latent_dim=int(_cfg.get("latent_dim", 64)),
                        num_resblocks=int(_cfg.get("num_resblocks", 6)))
    vnet.load_state_dict(vk["state_dict"])
    vsd = vnet.state_dict()

    if "value_state" not in ck:
        print("[graft] ERROR: ckpt has no value_state")
        return 1
    target = ck["value_state"]
    missing, mismatched = [], []
    for k, v in vsd.items():
        if k not in target:
            missing.append(k)
        elif tuple(target[k].shape) != tuple(v.shape):
            mismatched.append((k, tuple(target[k].shape), tuple(v.shape)))
        else:
            target[k] = v.clone()
    ck["value_state"] = target
    # ⚠️ 关键：ValueNetwork 是在**归一化标签** ((y-mean)/std) 上训练的，
    # 而 make_three_net_fn **不做反归一化**。所以把 y_std/y_mean 折进 value_head 最后一层：
    #   out_new = out_old * y_std + y_mean
    # 否则 terminal 标签会放大 1/0.386≈2.6 倍，rel 标签会放大 1/0.076≈13 倍（后者致命）。
    y_std = float(vk.get("y_std", 1.0))
    y_mean = float(vk.get("y_mean", 0.0))
    wkey, bkey = "value_head.2.weight", "value_head.2.bias"
    if wkey in target and bkey in target:
        target[wkey] = target[wkey] * y_std
        target[bkey] = target[bkey] * y_std + y_mean
        print(f"[graft] folded denorm into {wkey}: y_std={y_std:.5f} y_mean={y_mean:.5f}")
    else:
        print("[graft] WARNING: value_head.2 keys not found, denorm NOT folded")
    ck["value_graft"] = {"source": str(vp), "label": vk.get("label"),
                         "val_r2": vk.get("val_r2"), "seed": vk.get("seed"),
                         "y_mean": vk.get("y_mean"), "y_std": vk.get("y_std")}
    torch.save(ck, out)
    print(f"[graft] copied {len(vsd) - len(missing) - len(mismatched)}/{len(vsd)} tensors")
    if missing:
        print("[graft] missing keys:", missing[:8])
    if mismatched:
        print("[graft] shape mismatch:", mismatched[:4])
    print(f"[graft] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
