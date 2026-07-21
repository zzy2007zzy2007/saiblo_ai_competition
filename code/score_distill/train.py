"""Train a model to predict ActionCatalog scores from board state.

Phase 1 of the two-phase score regression pipeline:
    Phase 1 (this script): distill ActionCatalog scores into a model
    Phase 2 (ga_ss_train): use the pretrained model as initial checkpoint

Usage:
    # 1. Collect data first
    python code/score_distill/collect.py --games 1000 --workers 12

    # 2. Train score-pretrained model
    python code/score_distill/train.py --data-dir score_data --epochs 50
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
for p in (_REPO, Path(__file__).resolve().parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse, glob
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ScoreDataset(Dataset):
    """Dataset for score regression: board + stats → class_scores + score_map."""

    def __init__(self, data_dir: str, max_files: int | None = None):
        paths = sorted(glob.glob(str(Path(data_dir) / "chunk_*.npz")))
        if max_files:
            paths = paths[:max_files]

        boards, statss, scores, maps = [], [], [], []
        for p in paths:
            d = np.load(p)
            boards.append(d["board"])
            statss.append(d["stats"])
            scores.append(d["class_scores"])
            maps.append(d["score_map"])

        # Keep as float16 in RAM, convert to float32 in __getitem__ (saves ~50% memory)
        self.board = np.concatenate(boards, axis=0)        # (T, 28, 19, 19) float16
        self.stats = np.concatenate(statss, axis=0)        # (T, 42) float16
        self.class_scores = np.concatenate(scores, axis=0)  # (T, 24) float32
        self.score_map = np.concatenate(maps, axis=0)      # (T, 24, 19, 19) float16

        # Per-class z-score normalization (each class mean=0, std=1 across frames)
        for c in range(24):
            sc = self.class_scores[:, c].astype(np.float64)
            m, s = sc.mean(), sc.std()
            if s > 1e-8:
                self.class_scores[:, c] = ((sc - m) / s).astype(np.float32)
            mp = self.score_map[:, c].astype(np.float64).reshape(len(self), -1)
            m2, s2 = mp.mean(), mp.std()
            if s2 > 1e-8:
                self.score_map[:, c] = ((mp - m2) / s2).astype(np.float16).reshape(self.score_map[:, c].shape)

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        return {
            "board": torch.from_numpy(self.board[idx].astype(np.float32)),
            "stats": torch.from_numpy(self.stats[idx].astype(np.float32)),
            "class_scores": torch.from_numpy(self.class_scores[idx]),
            "score_map": torch.from_numpy(self.score_map[idx].astype(np.float32)),
        }


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/score_data")
    parser.add_argument("--max-files", type=int, default=None,
                        help="limit number of npz files to load")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="output directory (default: training_history/score_distill_YYYYMMDD_HHMMSS)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="save checkpoint every N epochs")
    args = parser.parse_args()

    # Output directory
    from datetime import datetime
    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("training_history") / f"score_distill_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # Load dataset
    dataset = ScoreDataset(args.data_dir, max_files=args.max_files)
    print(f"Loaded {len(dataset)} frames from {args.data_dir}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True)

    # Create model
    from my_ai.network import create_model
    model = create_model(num_heads=args.num_heads, small=args.small)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        total_score_loss = 0.0
        total_map_loss = 0.0
        n_batches = 0

        for batch in loader:
            board = batch["board"].to(device)           # (B, 28, 19, 19)
            stats = batch["stats"].to(device)           # (B, 42)
            target_scores = batch["class_scores"].to(device)  # (B, 24)
            target_map = batch["score_map"].to(device)  # (B, 24, 19, 19)

            optimizer.zero_grad()
            output = model(board, stats)

            # Score loss: MSE on softmax-normalized class scores
            B = board.size(0)
            score_dist = F.softmax(target_scores, dim=-1).detach()  # (B, 24)
            score_loss = 0.0
            for i in range(model.num_heads):
                logits = output[f"head{i+1}_logits"]  # (B, 24)
                pred_dist = F.log_softmax(logits, dim=-1)
                score_loss += F.kl_div(pred_dist, score_dist, reduction="batchmean")
            score_loss /= model.num_heads

            # Position loss: MSE on action_map vs score_map
            map_dist = F.softmax(target_map.reshape(B, 24, -1), dim=-1)  # (B, 24, 361)
            pred_map = F.log_softmax(output["action_map"].reshape(B, 24, -1), dim=-1)
            map_loss = F.kl_div(pred_map, map_dist.detach(), reduction="batchmean")

            loss = score_loss + map_loss
            loss.backward()
            optimizer.step()

            total_score_loss += score_loss.item()
            total_map_loss += map_loss.item()
            n_batches += 1

        print(f"epoch {epoch+1:3d}/{args.epochs}  "
              f"score_loss={total_score_loss/n_batches:.4f}  "
              f"map_loss={total_map_loss/n_batches:.4f}")

        # Save intermediate checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.cpu()
            ckpt_path = out_dir / f"epoch_{epoch+1:04d}.pt"
            torch.save({
                "model_state": model.state_dict(),
                "mean": model.get_parameters_as_vector(),
                "num_heads": model.num_heads,
                "config": vars(args),
            }, ckpt_path)
            model.to(device)
            print(f"  checkpoint -> {ckpt_path}")

    # Save final checkpoint
    model.cpu()
    final_path = out_dir / "final.pt"
    torch.save({
        "model_state": model.state_dict(),
        "mean": model.get_parameters_as_vector(),
        "num_heads": model.num_heads,
        "config": vars(args),
    }, final_path)
    print(f"\nFinal: {final_path}")
    print(f"Params: {model.count_parameters():,}")
    print(f"\nUse for GA training:")
    print(f"  python code/my_ai/ga_ss_train.py --checkpoint {final_path}")


if __name__ == "__main__":
    train()
