"""Train action class prediction head via behavior cloning.

Usage:
    # 1. Collect data first
    python code/bc/collect_data.py --games 500 --workers 12

    # 2. Train (default: freeze encoder, train only class head)
    python code/bc/train_bc.py --data bc_data_20260630_*.npz --epochs 20

    # 3. Optionally unfreeze encoder and fine-tune all params
    python code/bc/train_bc.py --data bc_data_20260630_*.npz --epochs 10 --no-freeze-encoder
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from my_ai.network import create_model


def load_data(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load .npz and return (board, stats, class_label) tensors."""
    data = np.load(path)
    board = torch.from_numpy(data["board"])  # (N, 28, 19, 19)
    stats = torch.from_numpy(data["stats"])  # (N, 42)
    label = torch.from_numpy(data["class_label"]).long()  # (N,)
    return board, stats, label


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Train BC class head")
    parser.add_argument("--data", type=str, required=True, help="path to .npz data file")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-freeze-encoder", action="store_true",
                        help="train all parameters (not just class head)")
    parser.add_argument("--out", type=str, default="bc_checkpoint.pt",
                        help="output checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────
    print(f"Loading data from {args.data}...")
    boards, stats, labels = load_data(args.data)
    N = len(labels)
    print(f"  {N:,} samples, board={tuple(boards.shape)}, stats={tuple(stats.shape)}")

    # Split
    perm = torch.randperm(N)
    split = int(N * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]

    train_boards = boards[train_idx].to(device)
    train_stats = stats[train_idx].to(device)
    train_labels = labels[train_idx].to(device)
    val_boards = boards[val_idx].to(device)
    val_stats = stats[val_idx].to(device)
    val_labels = labels[val_idx].to(device)

    # ── Model ────────────────────────────────────────────────────
    model = create_model(single_head=True).to(device)
    model.train()

    # Freeze encoder if requested
    if not args.no_freeze_encoder:
        for name, param in model.named_parameters():
            # freeze everything except policy_base and policy_head1
            if "policy_base" not in name and "policy_head1" not in name:
                param.requires_grad = False

    # Count trainable vs frozen parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable params: {trainable:,}  Frozen params: {frozen:,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ────────────────────────────────────────────
    best_val_acc = 0.0
    t0 = time.time()
    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'Time':>6}")

    for epoch in range(args.epochs):
        model.train()
        # Shuffle training data
        perm_train = torch.randperm(len(train_labels))
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for i in range(0, len(train_labels), args.batch_size):
            idx = perm_train[i:i + args.batch_size]
            b = train_boards[idx]
            s = train_stats[idx]
            lbl = train_labels[idx]

            out = model(b, s)
            logits = out["head1_logits"]
            loss = criterion(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc += compute_accuracy(logits.detach(), lbl)
            n_batches += 1

        train_loss = total_loss / n_batches
        train_acc = total_acc / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            out_val = model(val_boards, val_stats)
            val_logits = out_val["head1_logits"]
            val_loss = criterion(val_logits, val_labels).item()
            val_acc = compute_accuracy(val_logits, val_labels)

        elapsed = time.time() - t0
        print(f"{epoch:5d}  {train_loss:10.4f}  {train_acc:9.4f}  {val_loss:8.4f}  {val_acc:7.4f}  {elapsed:6.1f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save checkpoint
            mean = model.get_parameters_as_vector()
            torch.save({
                "mean": torch.from_numpy(mean),
                "model_state": model.state_dict(),
                "generation": -1,  # BC checkpoint marker
                "val_acc": val_acc,
            }, args.out)
            print(f"  → saved {args.out} (val_acc={val_acc:.4f})")

    print(f"\nDone. Best val_acc: {best_val_acc:.4f}  Checkpoint: {args.out}")

    # Print class distribution
    if torch.cuda.is_available():
        cpu_labels = labels.cpu()
    else:
        cpu_labels = labels
    dist = torch.bincount(cpu_labels).tolist()
    print(f"Class distribution: {dist}")
    print(f"Class entropy: {len([c for c in dist if c > 0])} / 23 classes present")


if __name__ == "__main__":
    main()
