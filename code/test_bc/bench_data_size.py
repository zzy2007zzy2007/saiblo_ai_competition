"""Experiment: how much elite data is enough for BC training?

Tests different (k, data_games) combinations to see if smaller data
can achieve the same loss convergence.
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
import torch
from torch.utils.data import DataLoader

from my_ai.network import create_model
from my_ai.ss_train import SSDataset, ss_supervised_update


def load_latest_checkpoint():
    """Find the latest ss_train checkpoint with bc_data."""
    ckpt_list = sorted(Path("training_history").glob("ss_20260703_*/gen_00*.pt"))
    if not ckpt_list:
        ckpt_list = sorted(Path("training_history").glob("ss_*/gen_00*.pt"))
    if not (ckpt_list and ckpt_list[-1].parent.joinpath("bc_data").exists()):
        return None, None, None

    ckpt_path = ckpt_list[-1]
    bc_dir = ckpt_path.parent / "bc_data"

    print(f"Checkpoint: {ckpt_path}")
    print(f"bc_data:    {bc_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    num_heads = ckpt.get("num_heads", 8)
    model = create_model(num_heads=num_heads)
    model.set_parameters_from_vector(ckpt["mean"].numpy())
    model.eval()
    return model, bc_dir, num_heads


def count_timesteps_per_ind(npz_paths, k_per_ind):
    """Estimate how many timesteps per individual."""
    # bc_data filenames: gen_{gen:04d}_ind{ind:03d}_seed{seed}.npz
    ind_sizes = {}
    for p in npz_paths:
        parts = p.stem.split("_")
        # gen_0000_ind000_seed0.npz → parts[2] = "ind000"
        ind_str = parts[2]
        data = np.load(p)
        ind_sizes[ind_str] = len(data["board"])
    return ind_sizes


def main():
    print("=" * 60)
    print("EXPERIMENT: BC training data size requirements")
    print("=" * 60)

    model, bc_dir, num_heads = load_latest_checkpoint()
    if model is None or bc_dir is None:
        print("No checkpoint with bc_data found!")
        return

    npz_paths = sorted(bc_dir.glob("*.npz"))
    print(f"Total .npz files: {len(npz_paths)}")

    # Count timesteps per individual
    ind_sizes = count_timesteps_per_ind(npz_paths, 0)
    print(f"Files per individual: {len(npz_paths) // max(len(set(ind_sizes)), 1)}")
    if ind_sizes:
        avg_ts = np.mean(list(ind_sizes.values()))
        print(f"Avg timesteps per file: {avg_ts:.0f}")
        print(f"Total timesteps: {sum(ind_sizes.values())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Test configurations ──
    # Simulate different k and data_games by subsetting .npz files
    configs = [
        ("k=3, data-games=6", 3, 6),
        ("k=3, data-games=18", 3, 18),
        ("k=5, data-games=18", 5, 18),  # current default
        ("k=1, data-games=6", 1, 6),
        ("k=2, data-games=6", 2, 6),
    ]

    # Group .npz by individual index
    from collections import defaultdict
    ind_files = defaultdict(list)
    for p in npz_paths:
        parts = p.stem.split("_")
        ind_str = parts[2]
        ind_files[ind_str].append(p)

    print(f"\nUnique individuals in bc_data: {len(ind_files)}")
    print("-" * 60)

    for label, k, data_games in configs:
        print(f"\n  Testing: {label}")

        # Simulate: pick top-K individuals, data_games files each
        # (in real training, we don't know which are top, so we just
        #  pick the first k individuals)
        selected_inds = sorted(ind_files.keys())[:k]
        selected_files = []
        for ind in selected_inds:
            files = sorted(ind_files[ind])
            selected_files.extend(files[:data_games])

        if not selected_files:
            print(f"    SKIP: no files for this config")
            continue

        # Build dataset
        ds = SSDataset(selected_files, p_hold=1.0)
        print(f"    Dataset: {len(ds)} samples from {len(selected_files)} files")

        if len(ds) == 0:
            print(f"    SKIP: empty dataset after HOLD filtering")
            continue

        # Train
        model_i = create_model(num_heads=num_heads)
        model_i.load_state_dict(model.state_dict())
        ret = ss_supervised_update(
            model_i, ds, device=device,
            epochs=3, lr=1e-3, batch_size=64,
        )
        print(f"    Result: cls={ret['class_loss']:.4f}  "
              f"map={ret['map_loss']:.4f}  "
              f"tot={ret['total_loss']:.4f}  "
              f"samples={ret['samples']}")

        # Check head preferences
        model_i.eval()
        model_i.to(device)
        loader = DataLoader(ds, batch_size=len(ds), shuffle=False)
        batch = next(iter(loader))
        board = batch["board"].to(device)
        stats = batch["stats"].to(device)
        with torch.no_grad():
            output = model_i(board, stats)
        head_div = []
        for hi in range(num_heads):
            logits = output[f"head{hi+1}_logits"]
            cls = logits.argmax(dim=-1).cpu().numpy()
            unique = len(set(cls))
            head_div.append(unique)
        cls_str = " ".join(f"H{h+1}={head_div[h]}" for h in range(num_heads))
        print(f"    Head unique classes: {cls_str}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
