"""Merge bc_data npz files per generation into single files for faster transfer.

Usage:
    # Merge a specific training output
    python code/tools/merge_bc_data.py training_history/ga_ss_20260721_134921/bc_data

    # Merge and then delete originals
    python code/tools/merge_bc_data.py training_history/ga_ss_20260721_134921/bc_data --delete
"""
from __future__ import annotations
import argparse, glob
from pathlib import Path
import numpy as np


def merge_bc_data(bc_dir: str, delete: bool = False):
    bc_dir = Path(bc_dir)
    # Group files by generation
    gen_patterns = set()
    for p in bc_dir.glob("gen_*_ind*.npz"):
        gen = p.stem.split("_ind")[0]  # "gen_0042"
        gen_patterns.add(gen)

    for gen in sorted(gen_patterns):
        paths = sorted(bc_dir.glob(f"{gen}_ind*.npz"))
        if not paths:
            continue

        boards, statss, classes, maps, logits, vals = [], [], [], [], [], []
        for p in paths:
            d = np.load(p)
            boards.append(d["board"])
            statss.append(d["stats"])
            classes.append(d["class_"])
            maps.append(d["action_map"])
            logits.append(d["head_logits"])
            if "value" in d:
                vals.append(d["value"])

        out = {
            "board": np.concatenate(boards),
            "stats": np.concatenate(statss),
            "class_": np.concatenate(classes),
            "action_map": np.concatenate(maps),
            "head_logits": np.concatenate(logits),
        }
        if vals:
            out["value"] = np.concatenate(vals)

        out_path = bc_dir / f"{gen}.npz"
        np.savez_compressed(out_path, **out)
        print(f"  {gen}: {len(paths)} files → {out_path.name}  ({len(boards[0])*len(paths):,} frames)")

        if delete:
            for p in paths:
                p.unlink()
            print(f"         deleted {len(paths)} original files")

    total_before = sum(1 for _ in bc_dir.glob("gen_*_ind*.npz"))
    total_after = sum(1 for _ in bc_dir.glob("gen_*.npz"))
    print(f"\nDone. {total_before} files → {total_after} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-generation npz files")
    parser.add_argument("bc_dir", type=str, help="path to bc_data directory")
    parser.add_argument("--delete", action="store_true", help="delete original files after merge")
    args = parser.parse_args()
    merge_bc_data(args.bc_dir, delete=args.delete)
