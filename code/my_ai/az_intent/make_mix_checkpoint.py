"""Mix a policy network and a value network into one split checkpoint.

对照试验工具：把任意两个 checkpoint 的策略网络（model_state）和价值网络
（value_state）拼成一个 split checkpoint，用于测试"策略来自 A、价值来自 B"
的组合强度（如 az_mix_r69p_v50v = r69 策略 + v50 训透价值）。

用法：
  python code/my_ai/az_intent/make_mix_checkpoint.py \
      --policy ckpt_p.pt --value ckpt_v.pt --out out.pt

说明：
  - 策略网络取 --policy 的 model_state
  - 价值网络取 --value 的 value_state；若 --value 是单模型（无 value_state），
    则取它的 model_state（相当于用同一个网络充当价值头）
  - 元信息（num_heads/no_bn/latent_dim/num_resblocks）取 --policy 的
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

_META = ("num_heads", "no_bn", "latent_dim", "num_resblocks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix policy + value nets into one split checkpoint")
    parser.add_argument("--policy", required=True, help="checkpoint whose model_state is used as the policy net")
    parser.add_argument("--value", required=True, help="checkpoint whose value_state (or model_state) is the value net")
    parser.add_argument("--out", required=True, help="output .pt path")
    args = parser.parse_args()

    p = torch.load(args.policy, map_location="cpu", weights_only=False)
    v = torch.load(args.value, map_location="cpu", weights_only=False)
    if "model_state" not in p:
        raise SystemExit(f"--policy {args.policy} has no model_state")
    value_state = v.get("value_state", v.get("model_state"))
    if value_state is None:
        raise SystemExit(f"--value {args.value} has neither value_state nor model_state")

    mix = {"model_state": p["model_state"], "value_state": value_state}
    for k in _META:
        if k in p:
            mix[k] = p[k]
    # The value net may use a DIFFERENT normalization than the policy (e.g. a
    # GroupNorm value net with a BatchNorm policy) — record it so the loaders
    # build each net with its own config (see load_split_models).
    mix["value_no_bn"] = bool(v.get("no_bn", False))
    mix["value_gn"] = bool(v.get("gn", False))
    mix["value_gn_groups"] = int(v.get("gn_groups", p.get("gn_groups", 8)))
    mix["value_value_pool"] = str(v.get("value_pool", "gap"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mix, out)
    n_p = sum(t.numel() for t in p["model_state"].values())
    n_v = sum(t.numel() for t in value_state.values())
    print(f"[mix] policy={args.policy} -> model_state ({n_p:,} params)")
    print(f"[mix] value={args.value} -> value_state ({n_v:,} params)")
    print(f"[mix] saved -> {out}")
    print(f"[mix] metadata={ {k: mix[k] for k in _META if k in mix} }")


if __name__ == "__main__":
    main()
