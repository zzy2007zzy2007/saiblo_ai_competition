"""Train a ValueNetwork from collected game .npz data.

Usage:
    python code/my_ai/train_value_net.py
        --data-dirs distill_data_v2 distill_data_biased
        --epochs 50
        --lr 1e-3
        --batch-size 256
        --small
        --save value_net.pt

Data format:
    Each .npz file contains:
      - board:  (T, 28, 19, 19) float16
      - stats:  (T, 42)         float16
      - value:  (T, 2)          float32  — (hp_us, hp_opp) per frame

Label:
    value_target = (hp_us - hp_opp) / 20.0    # normalized HP difference
    Network output: bare scalar (no Tanh)
    Loss: MSE

The script scans all .npz files in the specified directories, loads
board/stats/value, computes the target, and trains the value network
via MSE regression.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _CODE_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import csv
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from my_ai.value_network import create_value_network, create_small_value_network


class ValueDataset(Dataset):
    """Dataset of (board, stats) → value label.

    Two label modes:
      - ``tau=None`` (default):  target = (hp_us - hp_opp) / 20.0
          Uses the final HP difference per game — works with any data format.
      - ``tau=N``:                target = (weighted_future_advantage - current_advantage) / 20.0
          Exponentially weighted future HP difference relative to current,
          computed per-game via backward pass.  Requires per-frame HP data
          (new collection format).
    """

    def __init__(self, npz_paths: list[Path], tau: float | None = None):
        print(f"  [ValueDataset] Loading {len(npz_paths)} files into memory (float16)...")
        t0 = time.perf_counter()
        if tau is not None:
            print(f"  [ValueDataset] tau={tau:.1f}  (exponentially weighted labels)")

        boards, statss, targets = [], [], []
        total = 0
        for i, p in enumerate(npz_paths):
            data = np.load(p)
            n = data["value"].shape[0]

            # Keep as float16 — saves ~50% memory vs float32
            boards.append(data["board"])          # (T, 28, 19, 19) float16
            statss.append(data["stats"])           # (T, 42) float16
            val = data["value"]                    # (T, 2) float32

            if tau is not None:
                # ── Exponentially weighted future advantage (relative) ──
                gamma = np.exp(-1.0 / tau)
                d = val[:, 0] - val[:, 1]          # (T,) per-frame HP diff
                labels = np.empty(n, dtype=np.float32)
                suffix_sum = 0.0
                weight_sum = 0.0
                for t in reversed(range(n)):
                    suffix_sum = d[t] + gamma * suffix_sum
                    weight_sum = 1.0 + gamma * weight_sum
                    raw = suffix_sum / weight_sum
                    labels[t] = (raw - d[t]) / 20.0
                targets.append(labels)
            else:
                # ── Original: normalised final HP difference ────────────
                hp_us = val[:, 0]
                hp_opp = val[:, 1]
                targets.append((hp_us - hp_opp) / 20.0)

            total += n
            if (i + 1) % 100 == 0 or i == len(npz_paths) - 1:
                mb = total * (28*19*19 + 42) * 2 / 1024 / 1024  # float16 bytes
                print(f"    Loaded {i+1}/{len(npz_paths)} files  ({total} frames, ~{mb:.0f}MB)", flush=True)

        # Concatenate (still float16 — ~2.7GB for 180K frames)
        self.board = np.concatenate(boards, axis=0)    # (N, 28, 19, 19) float16
        self.stats = np.concatenate(statss, axis=0)    # (N, 42) float16
        self.target = np.concatenate(targets, axis=0)  # (N,) float32

        dt = time.perf_counter() - t0
        mem_mb = (self.board.nbytes + self.stats.nbytes + self.target.nbytes) / 1024 / 1024
        print(f"  Done in {dt:.1f}s  ({len(self)} frames, {mem_mb:.0f}MB total)")
        print(f"    target range: [{self.target.min():.3f}, {self.target.max():.3f}]")
        print(f"    target mean:  {self.target.mean():.3f}")

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return {
            "board": torch.from_numpy(self.board[idx].astype(np.float32)),       # (28, 19, 19)
            "stats": torch.from_numpy(self.stats[idx].astype(np.float32)),       # (42,)
            "target": torch.tensor(self.target[idx], dtype=torch.float32),  # scalar
        }


def find_npz_files(data_dirs: list[Path]) -> list[Path]:
    """Recursively collect all .npz files from given directories."""
    paths = []
    for d in data_dirs:
        d = Path(d)
        if d.is_dir():
            paths.extend(sorted(d.rglob("*.npz")))
        else:
            print(f"  [WARN] data directory not found: {d}")
    return paths


@torch.no_grad()
def evaluate(model, loader, desc="eval"):
    """Compute mean absolute error over a DataLoader."""
    model.eval()
    total_mae = 0.0
    count = 0
    for batch in loader:
        board = batch["board"].cuda()
        stats = batch["stats"].cuda()
        target = batch["target"].cuda()
        pred = model(board, stats)
        total_mae += (pred - target).abs().sum().item()
        count += len(target)
    model.train()
    avg_mae = total_mae / max(count, 1)
    print(f"  [{desc}] MAE = {avg_mae:.4f}  "
          f"(value range: [{target.min().item():.2f}, {target.max().item():.2f}])")
    return avg_mae


def _run_normal(args, model, optimizer, scheduler, out_dir, csv_path, start_epoch, device):
    """Original non-streaming training: load all data into memory."""
    # ── Collect data ────────────────────────────────────────────────────
    data_dirs = [Path(d) for d in args.data_dirs]
    npz_paths = find_npz_files(data_dirs)
    print(f"\nFound {len(npz_paths)} .npz files")

    if len(npz_paths) == 0:
        print("ERROR: no .npz files found. Check --data-dirs.")
        sys.exit(1)

    # ── Build dataset ───────────────────────────────────────────────────
    t_data = time.perf_counter()
    dataset = ValueDataset(npz_paths, tau=args.tau)
    dt_data = time.perf_counter() - t_data
    print(f"  Data loading: {dt_data:.1f}s")

    # Optional subsample for quick testing
    if args.subsample > 0 and args.subsample < len(dataset):
        print(f"  Subsampling to {args.subsample} frames (from {len(dataset)})")
        indices = torch.randperm(len(dataset))[:args.subsample]
        dataset = torch.utils.data.Subset(dataset, indices)
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset[i]["target"])
        targets = torch.stack(targets)
        print(f"    target range: [{targets.min():.3f}, {targets.max():.3f}]")
        print(f"    target mean:  {targets.mean():.3f}")

    # Train / val split
    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # ── Training loop ───────────────────────────────────────────────────
    _training_loop(args, model, optimizer, scheduler, out_dir, csv_path,
                   start_epoch, train_loader, val_loader)


def _run_streaming(args, model, optimizer, scheduler, out_dir, csv_path, start_epoch, device):
    """Streaming training: load merged files one at a time, release after use.

    Data dirs should contain ``merged_*.npz`` files produced by
    ``merge_value_npz.py``.  Each file has ``board``, ``stats``, ``target`` keys
    with pre-computed tau labels.
    """
    # ── Find merged files ───────────────────────────────────────────────
    data_dirs = [Path(d) for d in args.data_dirs]
    merged_files: list[Path] = []
    for d in data_dirs:
        merged_files.extend(sorted(d.glob("merged_*.npz")))
    if not merged_files:
        print("ERROR: no merged_*.npz files found in --data-dirs.  "
              "Run merge_value_npz.py first.")
        sys.exit(1)
    print(f"\nFound {len(merged_files)} merged files")

    # ── Hold out one file for validation ────────────────────────────────
    random.Random(args.seed).shuffle(merged_files)
    val_file = merged_files[-1]
    train_files = merged_files[:-1]
    # Estimate total frames: ~45K per merged file
    n_train_frames_est = len(train_files) * 45000
    print(f"  train: {len(train_files)} files (~{n_train_frames_est:,} frames)")

    print(f"  Loading val file: {val_file.name}...", flush=True)
    val_data = np.load(val_file)
    val_board = torch.from_numpy(val_data["board"].astype(np.float32))   # keep on CPU
    val_stats = torch.from_numpy(val_data["stats"].astype(np.float32))
    val_target = torch.from_numpy(val_data["target"])
    print(f"  val frames: {val_board.shape[0]} (kept on CPU, batched during eval)")

    # ── Load val data once (fits in memory, just one file) ──────────────
    val_data = np.load(val_file)
    val_board = torch.from_numpy(val_data["board"].astype(np.float32)).to(device)
    val_stats = torch.from_numpy(val_data["stats"].astype(np.float32)).to(device)
    val_target = torch.from_numpy(val_data["target"]).to(device)
    print(f"  val frames: {val_board.shape[0]}")

    # ── Training loop ───────────────────────────────────────────────────
    best_mae = float("inf")
    best_state = None
    n_batches_per_file = len(train_files) * (n_train_frames_est // len(train_files) // args.batch_size + 1)
    print(f"\n  ~{n_batches_per_file} batches per epoch, {args.epochs} epochs total\n")
    epoch_times: list[float] = []

    if start_epoch > 1:
        for _ in range(start_epoch - 1):
            scheduler.step()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t_epoch = time.perf_counter()

        random.Random(args.seed + epoch).shuffle(train_files)

        for fi, fpath in enumerate(train_files):
            # ── Load one merged file ────────────────────────────────────
            data = np.load(fpath)
            f_board = torch.from_numpy(data["board"].astype(np.float32))
            f_stats = torch.from_numpy(data["stats"].astype(np.float32))
            f_target = torch.from_numpy(data["target"])
            del data

            # Mini DataLoader for this file
            from torch.utils.data import TensorDataset
            ds = TensorDataset(f_board, f_stats, f_target)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
            del ds, f_board, f_stats, f_target

            for batch in loader:
                board, stats, target = [x.to(device) for x in batch]
                pred = model(board, stats)
                loss = nn.functional.mse_loss(pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if (fi + 1) % 5 == 0 or fi == 0:
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}] [epoch {epoch}/{args.epochs}] file {fi+1}/{len(train_files)}  "
                      f"loss={loss.item():.6f}  ({n_batches} batches)", flush=True)

            del loader  # release memory from this file

        # ── Validation ──────────────────────────────────────────────────
        ts = time.strftime("%H:%M:%S")
        print(f"  [{ts}] Validating...", flush=True)
        scheduler.step()
        epoch_dt = time.perf_counter() - t_epoch
        epoch_times.append(epoch_dt)

        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        val_mae = 0.0
        val_n = 0
        with torch.no_grad():
            for i in range(0, val_board.shape[0], args.batch_size):
                b = val_board[i:i+args.batch_size].to(device)
                s = val_stats[i:i+args.batch_size].to(device)
                t = val_target[i:i+args.batch_size].to(device)
                pred = model(b, s)
                val_mae += (pred.squeeze(-1) - t).abs().sum().item()
                val_n += t.shape[0]
        val_mae /= val_n
        model.train()

        # ETA
        avg_epoch_time = np.mean(epoch_times)
        remaining = (args.epochs - epoch) * avg_epoch_time
        eta_str = f"{remaining/60:.0f}m {remaining%60:.0f}s" if remaining < 3600 else f"{remaining/3600:.1f}h"

        if epoch == 1 or epoch % 5 == 0 or val_mae < best_mae:
            ts = time.strftime("%H:%M:%S")
            print(f"  [{ts}] epoch {epoch:3d}/{args.epochs}  train_loss={avg_loss:.6f}  "
                  f"val_mae={val_mae:.4f}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"[{epoch_dt:.0f}s/epoch, ETA {eta_str}]", flush=True)

        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  → new best model (MAE={best_mae:.4f})", flush=True)

        # ── CSV ─────────────────────────────────────────────────────────
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{avg_loss:.6f}", f"{val_mae:.4f}",
                        f"{scheduler.get_last_lr()[0]:.2e}", f"{epoch_dt:.1f}",
                        f"{best_mae:.4f}"])

        # ── Checkpoint ──────────────────────────────────────────────────
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = out_dir / f"epoch_{epoch:04d}.pt"
            torch.save({
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "best_mae": best_mae,
                "train_loss": avg_loss,
                "val_mae": val_mae,
                "params": model.count_parameters(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path}")

    # ── Save final best model ───────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state or model.state_dict(),
        "epoch": epoch,
        "params": model.count_parameters(),
        "best_mae": best_mae,
        "args": vars(args),
    }, save_path)
    print(f"\nSaved best model to {save_path}  (MAE={best_mae:.4f})")
    print(f"  Training history: {csv_path}")
    print(f"  Checkpoints:      {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train value network from .npz data")
    parser.add_argument("--data-dirs", nargs="+",
                        default=["distill_data_v2"],
                        help="Directories containing .npz files")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--small", action="store_true",
                        help="Use small value network (fewer params)")
    parser.add_argument("--save", type=str, default="value_net.pt",
                        help="Output model path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data used for validation")
    parser.add_argument("--subsample", type=int, default=0,
                        help="If >0, randomly sample N frames for quick testing")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs (0 = disable)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path (.pt file)")
    parser.add_argument("--tau", type=float, default=None,
                        help="Exponentially weighted future HP difference label "
                             "(tau = time horizon in turns).  Requires per-frame HP data "
                             "(new collection format).  Default: None = final HP diff label")
    parser.add_argument("--stream", action="store_true",
                        help="Streaming mode: load merged files one at a time. "
                             "Data dirs should contain merged_*.npz files.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Value Network Training")
    print(f"  Data dirs:   {args.data_dirs}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Weight decay:{args.weight_decay}")
    print(f"  Small model: {args.small}")
    print(f"  Label mode:  {'tau=' + str(args.tau) + ' (exponential weighted)' if args.tau else 'final HP diff'}")
    print(f"  Save path:   {args.save}")
    print(f"  Streaming:   {'yes' if args.stream else 'no'}")
    print("=" * 60)

    # ── Create model ────────────────────────────────────────────────────
    if args.small:
        model = create_small_value_network()
    else:
        model = create_value_network()
    print(f"Model parameters: {model.count_parameters():,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = model.cuda() if device == "cuda" else model

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Setup output directory ────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("training_history") / f"value_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output: {out_dir}/")

    # CSV history
    csv_path = out_dir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_mae", "lr", "epoch_time_s", "best_mae"])

    # Resume from checkpoint
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  Resumed from {args.resume}  (epoch {start_epoch-1} → resume at epoch {start_epoch})")

    if args.stream:
        _run_streaming(args, model, optimizer, scheduler, out_dir, csv_path, start_epoch, device)
    else:
        _run_normal(args, model, optimizer, scheduler, out_dir, csv_path, start_epoch, device)


if __name__ == "__main__":
    main()
