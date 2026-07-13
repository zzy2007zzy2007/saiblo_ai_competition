"""Combined diagnosis: behavior + head analysis in one shot.

Usage:
    python code/test_match/diagnose.py <checkpoint.pt>

Runs:
    1. diagnose_model (1 game, verbose) — what actions the model takes
    2. diagnose_heads (1 game) — what each head wants to do
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from utils.logger import get_logger
from diagnose_model import diagnose
from diagnose_heads import diagnose_heads


def main():
    import argparse

    torch.set_num_threads(1)
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")

    parser = argparse.ArgumentParser(description="Combined diagnosis: behavior + head analysis")
    parser.add_argument("ckpt", type=str, help="checkpoint path")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--action-dropout", type=float, default=0.0, help="action dropout rate (simulate training condition)")
    parser.add_argument("--ind", type=int, default=None,
                        help="population index to diagnose (default: TOP1/mean)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="log directory (dual output to terminal + file)")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Auto-detect num_heads
    if "num_heads" in ckpt:
        num_heads = ckpt["num_heads"]
    elif "config" in ckpt and "num_heads" in ckpt["config"]:
        num_heads = ckpt["config"]["num_heads"]
    else:
        import re
        hk = [k for k in ckpt.get("model_state", ckpt) if re.match(r"policy_heads\.\d+\.weight", k)]
        num_heads = max(len(hk), 1) if hk else 3

    log = None
    if args.log_dir:
        log = get_logger(Path(args.log_dir) / "diagnose.log", mode="a")

    p = (lambda *a, **kw: log.print(*a, **kw)) if log else print
    p("=" * 60)
    p(f"DIAGNOSE: {args.ckpt}")
    p(f"num_heads={num_heads}")
    p("=" * 60)

    # 1. Behavior diagnosis (1 game verbose)
    p("\n" + "-" * 60)
    p("BEHAVIOR vs ExampleAI")
    p("-" * 60)
    diagnose(args.ckpt, n_games=1, seed_offset=args.seed, verbose=True, num_heads=num_heads, log=log, action_dropout=args.action_dropout, ind=args.ind)

    # 2. Head diagnosis (1 game)
    p("\n" + "-" * 60)
    p("HEAD ANALYSIS vs ExampleAI")
    p("-" * 60)
    diagnose_heads(args.ckpt, seed=args.seed, num_heads=num_heads, log=log, action_dropout=args.action_dropout, ind=args.ind)


if __name__ == "__main__":
    main()
