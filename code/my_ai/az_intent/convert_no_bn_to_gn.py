"""Convert a no_bn checkpoint to the equivalent GroupNorm model.

Why GroupNorm (docs/az_groupnorm_backbone_plan.md):
  - no_bn  → activation scale uncontrolled → feature collapse under training (§20)
  - BatchNorm → running stats drift → training and inference become different
    functions (measured: same data + frozen backbone, BN 9.4% vs no_bn 28.1%)
  - GroupNorm → per-sample statistics, NO running buffers → training == inference,
    while still controlling activation scale → the backbone can be trained safely.

Migration (same bias-shift trick as convert_no_bn_to_bn):
  conv.weight stays; conv.bias → GroupNorm beta; GroupNorm gamma = 1.
NOTE: unlike the BN migration this is NOT output-equivalent (GroupNorm normalises
per sample, which changes the function) — retrain after converting.

Usage:
    python code/my_ai/az_intent/convert_no_bn_to_gn.py \
        --input training_history/az_intent/gen0120_warm_cpp.pt \
        --output training_history/az_intent/gen0120_warm_cpp_gn.pt [--gn-groups 8]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
import sys
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from my_ai.network import create_model  # noqa: E402


def _migrate_state(src_state: dict) -> dict:
    """no_bn structure → GroupNorm structure (conv.bias → GN beta, gamma=1)."""
    out: dict = {}
    for key, val in src_state.items():
        if key.endswith(".bias") and (".conv1." in key or ".conv2." in key):
            continue  # moved into the following GN beta
        if key.startswith("initial_conv.") and key.endswith(".bias"):
            continue
        out[key] = val.clone()

    def _conv_to_gn(conv_key: str, gn_key: str):
        w = src_state[f"{conv_key}.weight"]
        bias = src_state[f"{conv_key}.bias"]
        C = w.shape[0]
        out[f"{conv_key}.weight"] = w.clone()
        out[f"{gn_key}.weight"] = torch.ones(C)   # gamma = 1
        out[f"{gn_key}.bias"] = bias.clone()      # beta = old conv bias

    for i in range(20):
        c1, c2 = f"resblocks.{i}.conv1", f"resblocks.{i}.conv2"
        if f"{c1}.weight" in src_state:
            _conv_to_gn(c1, f"resblocks.{i}.bn1")
            _conv_to_gn(c2, f"resblocks.{i}.bn2")
        else:
            break

    if "initial_conv.0.bias" in src_state:
        w = src_state["initial_conv.0.weight"]
        C = w.shape[0]
        out["initial_conv.0.weight"] = w.clone()
        out["initial_conv.1.weight"] = torch.ones(C)
        out["initial_conv.1.bias"] = src_state["initial_conv.0.bias"].clone()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate no_bn checkpoint to GroupNorm model")
    parser.add_argument("--input", required=True, help="source no_bn checkpoint (.pt)")
    parser.add_argument("--output", required=True, help="output GroupNorm checkpoint (.pt)")
    parser.add_argument("--gn-groups", type=int, default=8, help="GroupNorm channel groups")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    meta = {k: ckpt[k] for k in ("num_heads", "latent_dim", "num_resblocks") if k in ckpt}
    print(f"[convert] source: {args.input}  meta={meta} no_bn={ckpt.get('no_bn')} "
          f"gn_groups={args.gn_groups}")

    def _build():
        return create_model(num_resblocks=ckpt.get("num_resblocks", 6),
                            num_heads=ckpt.get("num_heads", 3),
                            latent_dim=ckpt.get("latent_dim", 64),
                            no_bn=False, gn=True, gn_groups=args.gn_groups)

    def _migrate_net(src_sd, net):
        net.load_state_dict(_migrate_state(src_sd), strict=True)
        return net

    out_ckpt: dict = {}
    if "value_state" in ckpt:
        from my_ai.az_intent.az_selfplay import load_split_models
        pm, vm = load_split_models(args.input)
        out_ckpt["model_state"] = _migrate_net(pm.state_dict(), _build()).state_dict()
        out_ckpt["value_state"] = _migrate_net(vm.state_dict(), _build()).state_dict()
    else:
        out_ckpt["model_state"] = _migrate_net(ckpt["model_state"], _build()).state_dict()

    for k in ("num_heads", "latent_dim", "num_resblocks", "completed_batches", "warmup"):
        if k in ckpt:
            out_ckpt[k] = ckpt[k]
    out_ckpt["no_bn"] = False
    out_ckpt["gn"] = True
    out_ckpt["gn_groups"] = args.gn_groups

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_ckpt, out)
    print(f"[convert] saved -> {out}  (no_bn=False gn=True gn_groups={args.gn_groups})")


if __name__ == "__main__":
    main()
