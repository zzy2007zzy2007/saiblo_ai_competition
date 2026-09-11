"""Convert a no_bn=True checkpoint to the equivalent no_bn=False (BatchNorm) model.

fixes the feature-extractor collapse (docs/az_fix_feature_collapse_bn_migration.md):
az training drove resblocks[0-2] activations to explode (max 143) and killed
resblocks[3]'s ReLU, making spatial_feat input-independent.  BatchNorm prevents
this — but the existing weights were trained without it.  This script migrates
the model losslessly: each conv's bias moves into the corresponding BN's beta,
BN is initialised as identity (gamma=1, running_mean=0, running_var=1), so the
converted model is bit-identical at eval before further training, then BN
kicks in during training to normalise activations.

Usage:
    python code/my_ai/az_intent/convert_no_bn_to_bn.py \
        --input training_history/az_intent/gen0120_warm_cpp.pt \
        --output training_history/az_intent/gen0120_warm_cpp_bn.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from my_ai.network import create_model, ResBlock  # noqa: E402


def _migrate_state(src_state: dict) -> dict:
    """Migrate one network's state_dict from no_bn structure to BN structure.

    no_bn ResBlock: conv1(bias) / conv2(bias), bn1/bn2 = Identity (no params).
    BN ResBlock:    conv1(bias=False) / conv2(bias=False), bn1/bn2 = BatchNorm2d.
    Mapping: conv.bias -> bn.beta; conv.weight stays; bn = identity init
    (gamma=1, running_mean=0, running_var=1) so eval output is bit-identical.
    """
    out: dict = {}
    for key, val in src_state.items():
        if key.endswith(".bias") and (".conv1." in key or ".conv2." in key):
            continue  # conv bias moved into the following BN beta below
        if key.startswith("initial_conv.") and key.endswith(".bias"):
            continue  # handled below
        out[key] = val.clone()

    def _conv_to_bn(conv_key: str, bn_key: str):
        w = src_state[f"{conv_key}.weight"]
        bias = src_state[f"{conv_key}.bias"]
        out[f"{conv_key}.weight"] = w.clone()
        C = w.shape[0]
        out[f"{bn_key}.weight"] = torch.ones(C)          # gamma = 1
        out[f"{bn_key}.bias"] = bias.clone()             # beta = old conv bias
        out[f"{bn_key}.running_mean"] = torch.zeros(C)   # identity
        out[f"{bn_key}.running_var"] = torch.ones(C)     # identity
        out[f"{bn_key}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)

    # each ResBlock: conv1->bn1, conv2->bn2
    for i in range(20):
        c1 = f"resblocks.{i}.conv1"
        c2 = f"resblocks.{i}.conv2"
        if f"{c1}.weight" in src_state:
            _conv_to_bn(c1, f"resblocks.{i}.bn1")
            _conv_to_bn(c2, f"resblocks.{i}.bn2")
        else:
            break

    # initial_conv: Sequential[Conv(bias=True), ReLU] -> [Conv(bias=False), BN, ReLU]
    if "initial_conv.0.bias" in src_state:
        w = src_state["initial_conv.0.weight"]
        bias = src_state["initial_conv.0.bias"]
        C = w.shape[0]
        out["initial_conv.0.weight"] = w.clone()
        out["initial_conv.1.weight"] = torch.ones(C)
        out["initial_conv.1.bias"] = bias.clone()
        out["initial_conv.1.running_mean"] = torch.zeros(C)
        out["initial_conv.1.running_var"] = torch.ones(C)
        out["initial_conv.1.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate no_bn checkpoint to BatchNorm model")
    parser.add_argument("--input", required=True, help="source no_bn checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="output BN checkpoint (.pt)")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    meta = {k: ckpt[k] for k in ("num_heads", "latent_dim", "num_resblocks") if k in ckpt}
    print(f"[convert] source: {args.input}  meta={meta} no_bn={ckpt.get('no_bn')}")

    def _build():
        return create_model(num_resblocks=ckpt.get("num_resblocks", 6),
                            num_heads=ckpt.get("num_heads", 3),
                            latent_dim=ckpt.get("latent_dim", 64),
                            no_bn=False,
                            value_pool=ckpt.get("value_pool", "gap"))

    def _migrate_net(src_sd, net):
        new_sd = _migrate_state(src_sd)
        net.load_state_dict(new_sd, strict=True)
        return net

    out_ckpt: dict = {}
    if "value_state" in ckpt:
        # split checkpoint: migrate policy + value
        from my_ai.az_intent.az_selfplay import load_split_models
        pm, vm = load_split_models(args.input)
        out_ckpt["model_state"] = _migrate_net(pm.state_dict(), _build()).state_dict()
        out_ckpt["value_state"] = _migrate_net(vm.state_dict(), _build()).state_dict()
    else:
        out_ckpt["model_state"] = _migrate_net(ckpt["model_state"], _build()).state_dict()
    for k in ("num_heads", "no_bn", "latent_dim", "num_resblocks", "completed_batches", "warmup"):
        if k in ckpt:
            out_ckpt[k] = ckpt[k]
    out_ckpt["no_bn"] = False

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out)
    print(f"[convert] saved -> {out}  (no_bn={out_ckpt['no_bn']})")


if __name__ == "__main__":
    main()
