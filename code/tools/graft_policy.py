"""把蒸馏出的**单模型** ckpt 嫁接到我们的**三网部署** ckpt 上（2026-10-18）。

部署端（`my_ai/az_intent/az_selfplay.load_three_models`）要的格式：
    {class_state, pos_state, value_state, three_net=True, num_heads=3, latent_dim=64,
     num_resblocks=6, no_bn, gn, gn_groups, ...}
而 `code/distill/train.py` 的产物是 `save_checkpoint` 格式：`{model_state, mean, num_heads, ...}`。

本工具：`class_state`/`pos_state` ← 蒸馏网的 `model_state`（同架构 ⇒ 键应一致），
`value_state` **保留原网**（本轮不动价值网，S2 再单独重训）。

用法:
    python code/tools/graft_policy.py --base checkpoints/posnet_A_k5_m32.pt \
        --policy checkpoints/distill_sdk_policy.pt --out checkpoints/grafted_sdk_policy.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="部署用三网 ckpt（提供 value_state 与元数据）")
    ap.add_argument("--policy", required=True, help="蒸馏出的单模型 ckpt（model_state）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--also-value", action="store_true",
                    help="同时用蒸馏网覆盖 value_state（默认**不**动价值网）")
    a = ap.parse_args()

    base = torch.load(a.base, map_location="cpu")
    pol = torch.load(a.policy, map_location="cpu")
    ps = pol["model_state"] if isinstance(pol, dict) and "model_state" in pol else pol
    ps = {k: v for k, v in ps.items() if hasattr(v, "shape")}
    cs = base["class_state"]
    print(f"部署 class_state 键数 = {len(cs)}   蒸馏 model_state 键数 = {len(ps)}")
    miss = [k for k in cs if k not in ps]
    extra = [k for k in ps if k not in cs]
    print(f"  蒸馏缺的键（{len(miss)}）: {miss[:6]}")
    print(f"  蒸馏多的键（{len(extra)}）: {extra[:6]}")

    targets = ["class_state", "pos_state"] + (["value_state"] if a.also_value else [])
    for dst in targets:
        m = dict(base[dst])
        n = 0
        for k in list(m.keys()):
            if k in ps and tuple(ps[k].shape) == tuple(m[k].shape):
                m[k] = ps[k]
                n += 1
        base[dst] = m
        print(f"  {dst}: 覆盖 {n}/{len(m)} 个张量")
        if n != len(m):
            print(f"    ⚠️ 只覆盖了部分 ⇒ 该网是'混合体'，读数解释要小心")
    base["grafted_from"] = str(a.policy)
    base["graft_note"] = ("class/pos from distilled policy; value kept from base"
                          if not a.also_value else "all three from distilled policy")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, out)
    print("已写出:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
