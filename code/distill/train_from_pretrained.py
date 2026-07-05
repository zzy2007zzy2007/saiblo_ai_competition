"""Train a model using a pre-trained encoder.

The encoder is pre-trained with board reconstruction (self-supervised),
so it already extracts useful state-dependent features and cannot take
the "constant embedding" shortcut.

Usage:
    python code/distill/train_from_pretrained.py distill_data_v2/ \\
        --checkpoint distill_data_v2/pretrained_encoder.pt \\
        --out distill_model.pt --epochs 30
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
from my_ai.network import create_model
from my_ai.ss_train import SSDataset, ss_supervised_update, save_checkpoint
from utils.logger import get_logger


def compute_class_weights(dataset: SSDataset, max_weight: float = 5.0) -> torch.Tensor | None:
    labels = dataset.class_label
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels) * labels.shape[1]
    class_weights = np.ones(24, dtype=np.float32)
    for cid, cnt in zip(unique, counts):
        freq = cnt / total
        if freq > 0:
            class_weights[int(cid)] = 1.0 / freq
    class_weights = np.clip(class_weights, None, max_weight)
    class_weights = class_weights / class_weights.mean()
    return torch.from_numpy(class_weights)


def main():
    parser = argparse.ArgumentParser(description="BC training from pre-trained encoder")
    parser.add_argument("data_dir", type=str, help="directory with .npz files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="pre-trained encoder checkpoint (.pt)")
    parser.add_argument("--out", type=str, default="distill_model.pt",
                        help="output checkpoint path")
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--p-hold", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bias-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-div", type=float, default=0.05)
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--soft-temperature", type=float, default=2.0)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--log", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log = None
    if args.log:
        log = get_logger(args.log)

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    npz_paths = sorted(data_dir.glob("distill_*.npz"))
    if not npz_paths:
        print(f"No .npz files found in {data_dir}")
        return
    print(f"Found {len(npz_paths)} .npz files")
    ds = SSDataset(npz_paths, p_hold=args.p_hold)
    print(f"  Total samples: {len(ds)}")

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(ds, max_weight=args.max_class_weight)
        print(f"  Class weights: min={class_weights.min():.3f}, "
              f"max={class_weights.max():.3f}, mean={class_weights.mean():.3f}")

    # ── Model ─────────────────────────────────────────────────────────
    model = create_model(num_heads=args.num_heads)

    # Load pre-trained encoder weights
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder_state = ckpt["encoder_state"]

    # Extract encoder parameter names from model state
    # Encoder params are: initial_conv.*, resblocks.*, stats_mlp.*
    model_state = model.state_dict()
    for key in model_state:
        if key.startswith("initial_conv") or key.startswith("resblocks"):
            if key in encoder_state:
                model_state[key] = encoder_state[key]
                print(f"  Loaded pre-trained: {key}")

    model.load_state_dict(model_state)
    print(f"Model: {model.count_parameters():,} params, {args.num_heads} heads "
          f"(encoder pre-trained)")

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nTraining: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    ret = ss_supervised_update(model, ds, device,
                               epochs=args.epochs, lr=args.lr,
                               batch_size=args.batch_size,
                               label_smoothing=args.label_smoothing,
                               weight_decay=args.weight_decay,
                               bias_decay=args.bias_decay,
                               lambda_div=args.lambda_div,
                               lambda_soft=args.lambda_soft,
                               soft_temperature=args.soft_temperature,
                               class_weights=class_weights,
                               log=log)

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    mean = model.get_parameters_as_vector()
    save_checkpoint(out_path, mean, model, 0)
    print(f"\nSaved: {out_path}")
    print(f"  cls={ret['class_loss']:.4f}  map={ret['map_loss']:.4f}  "
          f"tot={ret['total_loss']:.4f}  samples={ret['samples']}")


if __name__ == "__main__":
    main()
