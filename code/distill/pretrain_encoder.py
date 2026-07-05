"""Pre-train the encoder with board reconstruction (self-supervised).

Forces the encoder to preserve spatial information in the board embedding.
After pre-training, the encoder CANNOT take the "constant embedding" shortcut
because a constant embedding can't reconstruct varying board states.

Usage:
    python code/distill/pretrain_encoder.py distill_data_v2/ \\
        --out pretrained_encoder.pt --epochs 10 --batch-size 64
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

from my_ai.network import AntWarNetwork, create_model


class BoardDataset(Dataset):
    """Load board data from .npz files for reconstruction pretraining."""

    def __init__(self, npz_paths):
        boards = []
        for p in npz_paths:
            data = np.load(p)
            boards.append(data["board"])
        self.board = np.concatenate(boards, axis=0).astype(np.float32)
        print(f"  Loaded {len(self.board)} boards from {len(npz_paths)} files")

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        return {
            "board": torch.from_numpy(self.board[idx]),  # (28, 19, 19)
        }


class PretrainModel(nn.Module):
    """AntWarNetwork encoder + reconstruction decoder."""

    def __init__(self, encoder: AntWarNetwork):
        super().__init__()
        self.encoder = encoder
        # Reconstruction decoder: board_emb (64) → (28*19*19) → reshape
        self.reconstruct = nn.Linear(64, 28 * 19 * 19)

    def forward(self, board):
        # Run encoder (only uses the CNN part)
        x = self.encoder.initial_conv(board)
        for block in self.encoder.resblocks:
            x = block(x)
        board_emb = x.mean(dim=[2, 3])  # (B, 64)
        # Reconstruct
        recon = self.reconstruct(board_emb)  # (B, 28*361)
        recon = recon.view(-1, 28, 19, 19)  # (B, 28, 19, 19)
        return recon

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def main():
    parser = argparse.ArgumentParser(description="Pre-train encoder with board reconstruction")
    parser.add_argument("data_dir", type=str, help="directory with .npz files")
    parser.add_argument("--out", type=str, default="pretrained_encoder.pt",
                        help="output checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-resblocks", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    npz_paths = sorted(data_dir.glob("distill_*.npz"))
    if not npz_paths:
        print(f"No .npz files found in {data_dir}")
        return

    print(f"Found {len(npz_paths)} .npz files")
    ds = BoardDataset(npz_paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    # ── Model ─────────────────────────────────────────────────────────
    encoder = create_model(num_resblocks=args.num_resblocks, num_heads=3)
    model = PretrainModel(encoder)
    print(f"Model: {model.count_parameters():,} params")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\nPre-training: epochs={args.epochs}, lr={args.lr}, bs={args.batch_size}")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        n = 0
        for batch in loader:
            board = batch["board"].to(device)
            recon = model(board)
            loss = F.mse_loss(recon, board)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        avg_loss = total_loss / max(n, 1)
        print(f"  Epoch {epoch+1}/{args.epochs}: recon_loss={avg_loss:.6f}")

    # ── Save encoder only ─────────────────────────────────────────────
    out_path = Path(args.out)
    torch.save({
        "encoder_state": encoder.state_dict(),
        "num_heads": 3,
        "num_resblocks": args.num_resblocks,
    }, out_path)
    print(f"\nSaved encoder checkpoint: {out_path}")


if __name__ == "__main__":
    main()
