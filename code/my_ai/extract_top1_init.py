"""Extract top1 from a checkpoint and save as init checkpoint (mean = top1)."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import numpy as np


def extract_top1_as_init(ckpt_path: str, out_path: str | None = None):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    if "top2_params" not in ckpt:
        print("No top2_params in checkpoint. Use --top1 to save with top2_params.")
        return

    top1 = ckpt["top2_params"][0]  # tensor or numpy
    if isinstance(top1, np.ndarray):
        top1 = torch.from_numpy(top1)

    # Create model and load top1 parameters
    from my_ai.network import create_model
    num_heads = ckpt.get("num_heads", 3)  # new format
    if num_heads == 3 and "single_head" in ckpt.get("config", {}):
        num_heads = 1  # old format single_head=True
    model = create_model(num_heads=num_heads)
    model.set_parameters_from_vector(top1.numpy())

    data = {
        "mean": top1.clone(),
        "velocity": torch.zeros_like(top1),
        "model_state": model.state_dict(),
        "generation": 0,
        "top2_params": [top1.clone()],
        "top2_scores": [1.0],
    }

    if out_path is None:
        p = Path(ckpt_path)
        out_path = str(p.parent / f"{p.stem}_top1_init.pt")

    torch.save(data, out_path)
    print(f"Saved init checkpoint: {out_path}")
    print(f"  num_heads={num_heads}, params={len(top1):,}")
    print(f"  generation=0 (will start fresh training from top1)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    extract_top1_as_init(args.ckpt, args.out)
