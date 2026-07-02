"""Combined diagnosis: behavior + head analysis in one shot.

Usage:
    python code/test_match/diagnose.py <checkpoint.pt>

Runs:
    1. diagnose_model (1 game, verbose) — what actions the model takes
    2. diagnose_heads (1 game) — what each head wants to do
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
from diagnose_model import diagnose
from diagnose_heads import diagnose_heads


def main():
    import argparse

    # Set CPU threads to 1 if no GPU (avoids thread contention)
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    print(f"Device: {device}")

    parser = argparse.ArgumentParser(description="Combined diagnosis: behavior + head analysis")
    parser.add_argument("ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)

    # Auto-detect num_heads
    if "num_heads" in ckpt:
        num_heads = ckpt["num_heads"]
    elif "config" in ckpt and "num_heads" in ckpt["config"]:
        num_heads = ckpt["config"]["num_heads"]
    else:
        import re
        hk = [k for k in ckpt.get("model_state", ckpt) if re.match(r"policy_heads\.\d+\.weight", k)]
        num_heads = max(len(hk), 1) if hk else 3

    print("=" * 60)
    print(f"DIAGNOSE: {args.ckpt}")
    print(f"num_heads={num_heads}")
    print("=" * 60)

    # 1. Behavior diagnosis (1 game verbose)
    print("\n" + "-" * 60)
    print("BEHAVIOR vs ExampleAI")
    print("-" * 60)
    diagnose(args.ckpt, n_games=1, seed_offset=args.seed, verbose=True, num_heads=num_heads)

    # 2. Head diagnosis (1 game)
    print("\n" + "-" * 60)
    print("HEAD ANALYSIS vs ExampleAI")
    print("-" * 60)
    diagnose_heads(args.ckpt, seed=args.seed)


if __name__ == "__main__":
    main()
