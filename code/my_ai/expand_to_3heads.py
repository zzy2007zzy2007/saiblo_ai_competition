"""Expand a 1-head checkpoint to 3 heads by copying head1 params to head2/head3.

Usage:
    python code/my_ai/expand_to_3heads.py <checkpoint.pt> [output.pt]
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import numpy as np
from my_ai.network import create_model


def expand_checkpoint(ckpt_path: str, out_path: str | None = None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    no_bn = ckpt.get("no_bn", False)
    small = ckpt.get("config", {}).get("small", False) if "config" in ckpt else False

    # Create 1-head and 3-head models
    model_1h = create_model(num_heads=1, small=small, no_bn=no_bn)
    model_3h = create_model(num_heads=3, small=small, no_bn=no_bn)

    # Load weights into 1-head model
    if "model_state" in ckpt:
        model_1h.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model_1h.set_parameters_from_vector(ckpt["mean"].numpy())

    # Copy backbone + head1 → head2, head3
    sd_1h = model_1h.state_dict()
    sd_3h = model_3h.state_dict()

    with torch.no_grad():
        for name in sd_3h:
            if name in sd_1h:
                sd_3h[name].copy_(sd_1h[name])
            elif "policy_heads.1." in name or "policy_heads.2." in name:
                # Copy from head0
                src_name = name.replace("policy_heads.1.", "policy_heads.0.").replace("policy_heads.2.", "policy_heads.0.")
                sd_3h[name].copy_(sd_1h[src_name])
            else:
                print(f"  WARNING: no source for {name}")

    model_3h.load_state_dict(sd_3h)

    # Save
    out_data = {
        "mean": torch.from_numpy(model_3h.get_parameters_as_vector()),
        "model_state": model_3h.state_dict(),
        "generation": ckpt.get("generation", 0),
        "num_heads": 3,
        "no_bn": no_bn,
        "top2_params": [torch.from_numpy(model_3h.get_parameters_as_vector())],
        "top2_scores": [1.0],
    }
    if "ga_pop" in ckpt:
        # Also expand ga_pop individuals
        expanded_pop = []
        for p in ckpt["ga_pop"]:
            model_1h.set_parameters_from_vector(p)
            with torch.no_grad():
                for name in sd_3h:
                    if name in sd_1h:
                        sd_3h[name].copy_(sd_1h[name])
                    elif "policy_heads.1." in name or "policy_heads.2." in name:
                        src = name.replace("policy_heads.1.", "policy_heads.0.").replace("policy_heads.2.", "policy_heads.0.")
                        sd_3h[name].copy_(sd_1h[src])
            model_3h.load_state_dict(sd_3h)
            expanded_pop.append(model_3h.get_parameters_as_vector())
        out_data["ga_pop"] = expanded_pop
    if "config" in ckpt:
        cfg = dict(ckpt["config"])
        cfg["num_heads"] = 3
        out_data["config"] = cfg
    if "leaderboard" in ckpt:
        out_data["leaderboard"] = ckpt["leaderboard"]

    out = out_path or (Path(ckpt_path).stem + "_3heads.pt")
    torch.save(out_data, out)
    print(f"Expanded to 3 heads: {out}")
    print(f"  Params: 1-head={model_1h.count_parameters():,} → 3-head={model_3h.count_parameters():,}")
    print(f"  no_bn={no_bn}")
    print()
    print(f"Now use it with:")
    print(f"  python code/my_ai/ga_ss_train.py --checkpoint {out} --num-heads 3 ...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    expand_checkpoint(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
