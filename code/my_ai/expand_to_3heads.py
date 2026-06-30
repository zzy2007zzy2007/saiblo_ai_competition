"""Expand a single-head checkpoint to 3 heads by copying head1 to head2/head3.

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
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Create a 3-head model with the same architecture
    model_3h = create_model(single_head=False)

    # Create a single-head model and load the mean
    model_1h = create_model(single_head=True)
    model_1h.set_parameters_from_vector(ckpt["mean"].numpy())

    # Copy all weights from 1-head to 3-head model
    params_1h = dict(model_1h.named_parameters())
    params_3h = dict(model_3h.named_parameters())

    with torch.no_grad():
        for name, p3 in model_3h.named_parameters():
            if name in params_1h:
                # Copy from single-head model
                p3.data.copy_(params_1h[name].data)
            elif name.startswith("policy_head2") or name.startswith("policy_head3"):
                # Copy from head1
                head1_name = name.replace("head2", "head1").replace("head3", "head1")
                p3.data.copy_(params_1h[head1_name].data)
            else:
                # Should not happen: new parameter with no source
                print(f"  WARNING: no source for {name}, keeping random init")

    # Save as new checkpoint
    out = out_path or (Path(ckpt_path).stem + "_3heads.pt")
    torch.save({
        "mean": torch.from_numpy(model_3h.get_parameters_as_vector()),
        "model_state": model_3h.state_dict(),
        "generation": ckpt.get("generation", 0),
    }, out)
    print(f"Expanded to 3 heads: {out}")
    print(f"  Mean shape: {model_3h.get_parameters_as_vector().shape}")
    print(f"  Params: {model_3h.count_parameters():,}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    ckpt = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    expand_checkpoint(ckpt, out)
