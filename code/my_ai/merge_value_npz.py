"""Merge small .npz game files into larger chunks for streaming training.

Each input .npz contains frames from one game (board, stats, value).
This script merges N games per output file and pre-computes tau labels,
reducing file count from thousands to dozens for efficient streaming I/O.

Usage:
    # Default: merge 100 games per file, tau=10
    python code/my_ai/merge_value_npz.py value_data/ --output-dir value_merged

    # Custom games per file and tau
    python code/my_ai/merge_value_npz.py value_data/ --games-per-file 200 --tau 5

Output .npz format:
    board:   (sum_T, 28, 19, 19) float16
    stats:   (sum_T, 42)         float16
    target:  (sum_T,)            float32  — weighted future advantage / 20.0
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np


def _compute_tau_targets(values: np.ndarray, tau: float) -> np.ndarray:
    """Compute exponentially weighted future HP difference labels.

    Args:
        values: (T, 2) float32  — per-frame (hp_us, hp_opp)
        tau:    time horizon in turns

    Returns:
        (T,) float32  — label_t = weighted_future_advantage / 20.0
    """
    gamma = np.exp(-1.0 / tau)
    d = values[:, 0] - values[:, 1]  # (T,)
    targets = np.empty(d.shape[0], dtype=np.float32)
    suffix_sum = 0.0
    weight_sum = 0.0
    for t in reversed(range(d.shape[0])):
        suffix_sum = d[t] + gamma * suffix_sum
        weight_sum = 1.0 + gamma * weight_sum
        raw = suffix_sum / weight_sum
        targets[t] = raw / 20.0
    return targets


def main():
    parser = argparse.ArgumentParser(description="Merge small .npz files into larger chunks")
    parser.add_argument("input_dir", type=str, help="Directory with raw .npz files (one per game)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for merged files (default: <input_dir>_merged)")
    parser.add_argument("--games-per-file", type=int, default=100,
                        help="Number of games to merge per output file (default: 100)")
    parser.add_argument("--tau", type=float, default=10.0,
                        help="Time horizon for exponentially weighted labels (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling input files (default: 42)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(str(input_dir) + "_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect input files ────────────────────────────────────────────────
    npz_paths = sorted(input_dir.rglob("*.npz"))
    print(f"Found {len(npz_paths)} .npz files in {input_dir}/")
    if not npz_paths:
        print("ERROR: no .npz files found.")
        return

    # Shuffle so each merged file contains a mix of different games
    random.Random(args.seed).shuffle(npz_paths)

    gpfile = args.games_per_file
    n_chunks = (len(npz_paths) + gpfile - 1) // gpfile  # ceiling division
    print(f"Merging into {n_chunks} chunks ({gpfile} games per file)")
    print(f"  tau = {args.tau}")
    print(f"  Output: {output_dir}/\n")

    total_frames = 0
    t0 = time.perf_counter()

    for chunk_idx in range(n_chunks):
        chunk_paths = npz_paths[chunk_idx * gpfile : (chunk_idx + 1) * gpfile]
        t_chunk = time.perf_counter()

        boards, statss, targets = [], [], []
        chunk_frames = 0

        for file_idx, p in enumerate(chunk_paths):
            data = np.load(p)
            n = data["board"].shape[0]

            boards.append(data["board"])                      # (T, 28, 19, 19) float16
            statss.append(data["stats"])                      # (T, 42) float16
            targets.append(_compute_tau_targets(data["value"], args.tau))  # (T,) float32

            chunk_frames += n
            total_frames += n

        # ── Concatenate and save ─────────────────────────────────────
        out_path = output_dir / f"merged_{chunk_idx:04d}.npz"
        np.savez_compressed(
            out_path,
            board=np.concatenate(boards, axis=0),       # (sum_T, 28, 19, 19) float16
            stats=np.concatenate(statss, axis=0),        # (sum_T, 42) float16
            target=np.concatenate(targets, axis=0),      # (sum_T,) float32
        )

        dt_chunk = time.perf_counter() - t_chunk
        n_files = len(chunk_paths)
        mib = out_path.stat().st_size / 1024 / 1024
        print(f"  [{chunk_idx+1:3d}/{n_chunks}]  {n_files:3d} games  {chunk_frames:6d} frames  "
              f"{mib:.0f}MB  {dt_chunk:.1f}s  → {out_path.name}", flush=True)

    dt = time.perf_counter() - t0
    print(f"\nDone in {dt:.0f}s")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Output files: {n_chunks} files → {output_dir}/")


if __name__ == "__main__":
    main()
