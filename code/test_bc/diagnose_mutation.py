"""Diagnose: verify gradient-guided mutation produces behavioral diversity.

Tests:
  1. Generate N mutated variants from a base model
  2. For a fixed set of game states, record each head's preferred class
  3. Count how many variants differ from the base model's behavior
  4. Compare with pure parameter noise baseline
"""
from __future__ import annotations
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from my_ai.network import create_model
from my_ai.ss_train import bc_mutate_population, SSDataset, _extract_grad_vector


class FixedStatesDataset(Dataset):
    """Return the same batch of states every time."""
    def __init__(self, board, stats, class_label, head_logits):
        self.board = torch.from_numpy(board).float()
        self.stats = torch.from_numpy(stats).float()
        self.class_label = torch.from_numpy(class_label).long()
        self.head_logits = torch.from_numpy(head_logits).float()

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        return {
            "board": self.board[idx],
            "stats": self.stats[idx],
            "class_label": self.class_label[idx],
            "action_map": torch.zeros(len(self.board), 24, 19, 19),
            "head_logits": self.head_logits[idx],
        }


def get_head_classes(model, board, stats, num_heads):
    """For each head, return its argmax class for each timestep."""
    model.eval()
    with torch.no_grad():
        output = model(board, stats)
    results = {}
    for hi in range(num_heads):
        logits = output[f"head{hi+1}_logits"]  # (B, 24)
        classes = logits.argmax(dim=-1).cpu().numpy()  # (B,)
        results[f"H{hi+1}"] = classes
    return results


def count_diversity(variants, base_model, board, stats, num_heads):
    """Compare each variant's head classes against base model.

    Returns:
        n_different: how many variants differ from base
        frac_global: fraction of timesteps where any variant disagrees with base
    """
    base_classes = get_head_classes(base_model, board, stats, num_heads)

    n_different = 0
    total_timesteps = len(board)
    n_heads_changed = np.zeros(num_heads, dtype=int)
    head_disagreement = np.zeros(num_heads, dtype=float)

    for variant_model in variants:
        var_classes = get_head_classes(variant_model, board, stats, num_heads)
        different = False
        for hi in range(num_heads):
            disagree = (var_classes[f"H{hi+1}"] != base_classes[f"H{hi+1}"]).mean()
            head_disagreement[hi] += disagree
            n_heads_changed[hi] += (disagree > 0)
            if disagree > 0:
                different = True
        if different:
            n_different += 1

    return {
        "n_different": n_different,
        "frac_different": n_different / len(variants),
        "head_disagreement": head_disagreement / len(variants),
        "n_heads_ever_changed": n_heads_changed,
    }


def test_mutation_population_size():
    """What's the maximum pop_size we can test in reasonable time?"""
    pass  # benchmark only


def load_or_create_data():
    """Try loading a checkpoint, or create random model + synthetic data."""
    # Check for a real checkpoint
    ckpt_dirs = sorted(Path("training_history").glob("ss_20260703_*/gen_00*.pt"))
    if not ckpt_dirs:
        ckpt_dirs = sorted(Path("training_history").glob("ss_*/gen_00*.pt"))
    if ckpt_dirs:
        ckpt_path = ckpt_dirs[-1]  # latest
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        num_heads = ckpt.get("num_heads", 8)
        model = create_model(num_heads=num_heads)
        model.set_parameters_from_vector(ckpt["mean"].numpy())
        model.eval()

        # Try loading bc_data from same dir
        bc_dir = ckpt_path.parent / "bc_data"
        npz_paths = sorted(bc_dir.glob("*.npz"))
        print(f"  bc_data: {len(npz_paths)} files")
        if npz_paths:
            ds = SSDataset(npz_paths[:5], p_hold=1.0)
            if len(ds) > 0:
                print(f"  dataset: {len(ds)} samples")
                loader = DataLoader(ds, batch_size=64, shuffle=True)
                batch = next(iter(loader))
                return model, ds, batch, num_heads

    # Fallback: random model + synthetic
    print("No checkpoint found, using random model + synthetic data")
    num_heads = 8
    model = create_model(num_heads=num_heads)
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    model.eval()

    # Synthetic data
    n = 64
    board = np.random.randn(n, 28, 19, 19).astype(np.float16)
    stats = np.random.randn(n, 42).astype(np.float16)
    class_label = np.random.randint(0, 24, size=(n, num_heads)).astype(np.int64)
    head_logits = np.random.randn(n, num_heads, 24).astype(np.float16)

    ds = FixedStatesDataset(board, stats, class_label, head_logits)
    batch = {
        "board": ds.board,
        "stats": ds.stats,
        "class_label": ds.class_label,
        "head_logits": ds.head_logits,
    }
    return model, ds, batch, num_heads


def main():
    print("=" * 60)
    print("DIAGNOSE: Gradient-guided mutation diversity check")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_model, ds, batch, num_heads = load_or_create_data()
    board_fixed = batch["board"].to(device)
    stats_fixed = batch["stats"].to(device)

    base_classes = get_head_classes(base_model, board_fixed, stats_fixed, num_heads)

    print(f"\nBase model head preferences (over {len(board_fixed)} states):")
    for hi in range(num_heads):
        cls = base_classes[f"H{hi+1}"]
        unique, counts = np.unique(cls, return_counts=True)
        top3 = sorted(zip(unique, counts), key=lambda x: -x[1])[:3]
        top_str = " ".join(f"{c}({n})" for c, n in top3)
        print(f"  H{hi+1}: {top_str}")

    # ── Test 1: Gradient-guided mutation at different configs ──
    configs = [
        ("default (step=50, temp=3.0)", 0.1, 3.0, 50.0, 20),
        ("big step (step=100, temp=3.0)", 0.1, 3.0, 100.0, 20),
        ("small step (step=10, temp=3.0)", 0.1, 3.0, 10.0, 20),
        ("more dirs (step=50, dirs=50)", 0.1, 3.0, 50.0, 50),
        ("high temp (step=50, temp=30.0)", 0.1, 30.0, 50.0, 20),
        ("pure noise (sigma=0.01)", 0.0, 3.0, 0.0, 0),
    ]

    n_variants = 24
    print(f"\nTesting {n_variants} variants per config")
    print("-" * 60)

    for label, p_mut, temp, step, ndirs in configs:
        variants = []

        if ndirs == 0:
            # Pure parameter noise baseline
            mean_params = base_model.get_parameters_as_vector()
            model_template = create_model(num_heads=num_heads)
            rng = np.random.RandomState()
            for i in range(n_variants):
                noise = rng.randn(*mean_params.shape).astype(np.float32) * step
                params = mean_params + noise
                model_template.set_parameters_from_vector(params)
                model_template.eval()
                variants.append(model_template)
        else:
            ds_copy = ds
            params_list = bc_mutate_population(
                base_model, ds_copy, device, n_variants,
                p_mutate=p_mut, temperature=temp,
                mutation_step=step, sigma=0.0,
                n_dirs=ndirs, log=None,
            )
            model_template = create_model(num_heads=num_heads)
            for params in params_list:
                model_template.set_parameters_from_vector(params)
                model_template.eval()
                variants.append(model_template)

        stats = count_diversity(variants, base_model, board_fixed, stats_fixed, num_heads)
        print(f"\n  {label}:")
        print(f"    Variants changed: {stats['frac_different']:.0%} ({stats['n_different']}/{n_variants})")
        print(f"    Per-head disagreement rate:")
        for hi in range(num_heads):
            pct = stats['head_disagreement'][hi] * 100
            nch = stats['n_heads_ever_changed'][hi]
            print(f"      H{hi+1}: {pct:.1f}% disagreed, {nch}/{n_variants} ever changed")
        print(f"    Avg head disagreement: {stats['head_disagreement'].mean() * 100:.1f}%")

    # ── Test 2: Mean model behavior change across training ──
    print("\n" + "=" * 60)
    print("Note: if all configs produce 0% diversity, check:")
    print("1. Are the gradient directions meaningful? (temperature too high = random)")
    print("2. Is mutation_step large enough? (step=0.01 on normalized grad)")
    print("3. Are the batch data diverse enough?")
    print("=" * 60)


if __name__ == "__main__":
    main()
