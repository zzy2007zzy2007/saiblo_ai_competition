"""Analyze class distribution in distillation .npz data files.

Usage:
    python code/distill/analyze_data.py distill_data/
"""

from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

# Class names for display
CLASS_NAMES = [
    "Build_Basic", "Heavy", "Heavy+", "Ice", "Bewitch",
    "Quick", "Quick+", "Double", "Sniper", "Mortar",
    "Mortar+", "Pulse", "Missile", "Producer_Fast",
    "Producer_Siege", "Producer_Medic",
    "Downgrade",  # 16
    "Lightning", "EMP", "Deflector", "Evasion",  # 17-20
    "SpeedUp", "AntUp",  # 21-22
    "HOLD",  # 23
]

CLASS_SHORT = [
    "0Basic",        # 0  Build_Basic
    "1Heavy",        # 1  Heavy
    "11Heavy+",      # 2  Heavy+
    "12Ice",         # 3  Ice
    "13Bewitch",     # 4  Bewitch
    "2Quick",        # 5  Quick
    "21Quick+",      # 6  Quick+
    "22Double",      # 7  Double
    "23Sniper",      # 8  Sniper
    "3Mortar",       # 9  Mortar
    "31Mortar+",     # 10 Mortar+
    "32Pulse",       # 11 Pulse
    "33Missile",     # 12 Missile
    "41Producer+",   # 13 Producer+
    "42Siege",       # 14 Siege
    "43Medic",       # 15 Medic
    "13Downgrade",   # 16 Downgrade
    "21Lightning",   # 17 Lightning
    "22EMP",         # 18 EMP
    "23Deflector",   # 19 Deflector
    "24Evasion",     # 20 Evasion
    "31Speed",       # 21 SpeedUp
    "32Hp",          # 22 HpUp
    "HOLD",          # 23 HOLD
]


def analyze(npz_paths: list[Path]):
    """Analyze class distribution in .npz files."""
    if not npz_paths:
        print("No .npz files found.")
        return

    all_classes = []  # list of (T, N_heads) arrays
    all_n_heads = None
    n_files = 0
    n_total_turns = 0

    for p in npz_paths:
        data = np.load(p)
        classes = data["class_"]  # (T, N_heads)
        n_files += 1
        n_total_turns += len(classes)
        all_classes.append(classes)
        if all_n_heads is None:
            all_n_heads = classes.shape[1]
        else:
            assert classes.shape[1] == all_n_heads

    all_classes = np.concatenate(all_classes, axis=0)  # (T_total, N_heads)

    # Per-head stats
    print(f"Files: {n_files}")
    print(f"Total turns: {n_total_turns}")
    print(f"Number of heads: {all_n_heads}")
    print()

    for h in range(all_n_heads):
        head_classes = all_classes[:, h]
        unique, counts = np.unique(head_classes, return_counts=True)
        total = len(head_classes)
        n_non_hold = np.sum(head_classes != 23)
        n_hold = np.sum(head_classes == 23)

        print(f"=== Head {h+1} ===")
        print(f"  Non-HOLD: {n_non_hold}/{total} ({100*n_non_hold/total:.1f}%)")
        print(f"  HOLD:     {n_hold}/{total} ({100*n_hold/total:.1f}%)")
        print(f"  Class distribution (top 15):")
        # Sort by count descending
        sorted_idx = np.argsort(-counts)
        for idx in sorted_idx:
            cid = unique[idx]
            pct = 100 * counts[idx] / total
            bar = "█" * max(1, int(pct / 2))
            name = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else f"Class_{cid}"
            short = CLASS_SHORT[int(cid)] if int(cid) < len(CLASS_SHORT) else "?"
            print(f"    {short:4s} {int(cid):3d} {name:20s} {counts[idx]:6d} ({pct:5.1f}%) {bar}")
        print()

    # Diversity: how many turns have all 3 heads predicting different classes?
    if all_n_heads >= 3:
        same = np.all(all_classes[:, 0:1] == all_classes[:, 1:], axis=1)
        n_same = np.sum(same)
        print(f"=== Head Diversity ===")
        print(f"  All heads same class: {n_same}/{n_total_turns} ({100*n_same/n_total_turns:.1f}%)")
        print()

    # Per-file stats (show top 10 most/least diverse files)
    print(f"=== Per-file stats ===")
    file_stats = []
    for p in npz_paths:
        data = np.load(p)
        classes = data["class_"]  # (T, 3)
        total = len(classes)
        # HOLD if any head is HOLD? or all heads are HOLD? Use all heads HOLD
        n_hold = np.sum(np.all(classes == 23, axis=1))
        non_hold = total - n_hold
        file_stats.append((p.name, total, int(n_hold), int(non_hold)))
    file_stats.sort(key=lambda x: -x[1])  # sort by total turns

    # Show a few examples
    print(f"  {'File':20s} {'Turns':>6s} {'HOLD':>6s} {'Non-HOLD':>9s} {'%Non-HOLD':>9s}")
    for name, total, n_hold, non_hold in file_stats[:10]:
        pct = 100 * non_hold / max(total, 1)
        print(f"  {name:20s} {total:6d} {n_hold:6d} {non_hold:9d} {pct:8.1f}%")
    if len(file_stats) > 10:
        print(f"  ... ({len(file_stats) - 10} more files)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze distillation data distribution")
    parser.add_argument("data_dir", type=str, help="directory with .npz files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    npz_paths = sorted(data_dir.glob("distill_*.npz"))
    if not npz_paths:
        # Also try *.npz
        npz_paths = sorted(data_dir.glob("*.npz"))

    analyze(npz_paths)


if __name__ == "__main__":
    main()