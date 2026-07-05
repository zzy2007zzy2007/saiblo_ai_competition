"""BC training with frozen pre-trained encoder + head-only warm-up.

Phase 1: FREEZE encoder, train only policy heads (prevents collapse).
Phase 2: UNFREEZE everything, fine-tune with low LR.

This two-phase approach prevents the model from "unlearning" useful
state-dependent features during BC training.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from my_ai.network import create_model
from my_ai.ss_train import SSDataset, save_checkpoint
from utils.logger import get_logger


def compute_class_weights(dataset, max_weight=5.0):
    labels = dataset.class_label
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels) * labels.shape[1]
    cw = np.ones(24, dtype=np.float32)
    for cid, cnt in zip(unique, counts):
        freq = cnt / total
        if freq > 0:
            cw[int(cid)] = 1.0 / freq
    cw = np.clip(cw, None, max_weight)
    cw = cw / cw.mean()
    return torch.from_numpy(cw)


def train_epoch(model, loader, optimizer, device, cw, label_smoothing):
    """One epoch of CE training with label smoothing and class weights."""
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        board = batch["board"].to(device)
        stats = batch["stats"].to(device)
        cls_label = batch["class_label"].to(device)

        optimizer.zero_grad()
        output = model(board, stats)

        cls_loss = 0.0
        for i in range(model.num_heads):
            logits = output[f"head{i+1}_logits"]
            log_probs = F.log_softmax(logits, dim=-1)
            if label_smoothing > 0:
                smooth_target = torch.full_like(log_probs, label_smoothing / 23)
                smooth_target.scatter_(1, cls_label[:, i:i+1], 1.0 - label_smoothing)
            else:
                smooth_target = torch.zeros_like(log_probs)
                smooth_target.scatter_(1, cls_label[:, i:i+1], 1.0)

            if cw is not None:
                sample_weight = cw[cls_label[:, i]].unsqueeze(1)
                loss = -(smooth_target * log_probs * sample_weight).sum(dim=-1).mean()
            else:
                loss = -(smooth_target * log_probs).sum(dim=-1).mean()
            cls_loss += loss
        cls_loss /= model.num_heads
        cls_loss.backward()
        optimizer.step()
        total_loss += cls_loss.item()
        n += 1
    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="BC with frozen encoder + head warm-up")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="pre-trained encoder checkpoint")
    parser.add_argument("--out", type=str, default="distill_model.pt")
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--epochs-phase1", type=int, default=10,
                        help="epochs with frozen encoder (head-only training)")
    parser.add_argument("--epochs-phase2", type=int, default=20,
                        help="epochs with full fine-tuning")
    parser.add_argument("--lr-phase1", type=float, default=1e-2,
                        help="higher LR for head-only training")
    parser.add_argument("--lr-phase2", type=float, default=1e-4,
                        help="lower LR for full fine-tuning")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--p-hold", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-class-weight", type=float, default=5.0)
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    class_weights = compute_class_weights(ds, max_weight=args.max_class_weight)
    print(f"  Class weights: min={class_weights.min():.3f}, "
          f"max={class_weights.max():.3f}, mean={class_weights.mean():.3f}")
    cw = class_weights.to(device) if class_weights is not None else None

    # ── Model ─────────────────────────────────────────────────────────
    model = create_model(num_heads=args.num_heads)

    # Load pre-trained encoder
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder_state = ckpt["encoder_state"]
    model_state = model.state_dict()
    loaded = 0
    for key in model_state:
        if key.startswith(("initial_conv", "resblocks")):
            if key in encoder_state:
                model_state[key] = encoder_state[key]
                loaded += 1
    model.load_state_dict(model_state)
    model.to(device)
    print(f"Model: {model.count_parameters():,} params ({loaded} encoder keys loaded)")

    # ── Phase 1: Frozen encoder, head-only ────────────────────────────
    print(f"\nPhase 1: head-only training (encoder frozen), "
          f"{args.epochs_phase1} epochs, lr={args.lr_phase1}")
    for param in model.initial_conv.parameters():
        param.requires_grad = False
    for block in model.resblocks:
        for param in block.parameters():
            param.requires_grad = False

    # Only policy heads + policy_base + action_map_conv are trainable
    head_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr_phase1, weight_decay=1e-4)

    for epoch in range(args.epochs_phase1):
        loss = train_epoch(model, loader, head_optimizer, device, cw,
                           args.label_smoothing)
        print(f"  Phase 1 epoch {epoch+1}/{args.epochs_phase1}: cls_loss={loss:.4f}")

    # ── Phase 2: Full fine-tune ───────────────────────────────────────
    print(f"\nPhase 2: full fine-tune, {args.epochs_phase2} epochs, lr={args.lr_phase2}")
    for param in model.initial_conv.parameters():
        param.requires_grad = True
    for block in model.resblocks:
        for param in block.parameters():
            param.requires_grad = True

    full_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_phase2,
                                        weight_decay=1e-4)

    for epoch in range(args.epochs_phase2):
        loss = train_epoch(model, loader, full_optimizer, device, cw,
                           args.label_smoothing)
        print(f"  Phase 2 epoch {epoch+1}/{args.epochs_phase2}: cls_loss={loss:.4f}")

    # ── Save ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    mean = model.get_parameters_as_vector()
    save_checkpoint(out_path, mean, model, 0)
    print(f"\nSaved: {out_path}")
    print(f"  Final cls_loss: {loss:.4f}")


if __name__ == "__main__":
    main()
