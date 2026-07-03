"""Benchmark: BC mutation for N individuals.

Measures how long it takes to generate N mutated variants
from a trained model using lightweight fine-tuning.
"""
from __future__ import annotations
import sys, time, tempfile, shutil
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

from my_ai.network import create_model, AntWarNetwork


# ═══════════════════════════════════════════════════════════════
# Copy of ss_train's SSDataset (to avoid import dependency issues)
# ═══════════════════════════════════════════════════════════════

class MutateDataset(Dataset):
    """Minimal dataset for BC mutation: returns clean data, we mutate labels on the fly."""
    HOLD_CLASS = 23

    def __init__(self, npz_paths, batch_size=64):
        boards, statss, classes = [], [], []
        for p in npz_paths:
            data = np.load(p)
            boards.append(data["board"])
            statss.append(data["stats"])
            classes.append(data["class_"])
        self.board = np.concatenate(boards, axis=0)
        self.stats = np.concatenate(statss, axis=0)
        self.class_label = np.concatenate(classes, axis=0)
        self.batch_size = batch_size
        self.shuffle_indices = np.random.permutation(len(self))

    def __len__(self):
        return len(self.board)

    def __getitem__(self, idx):
        return {
            "board": torch.from_numpy(self.board[idx]).float(),
            "stats": torch.from_numpy(self.stats[idx]).float(),
            "class_label": torch.from_numpy(self.class_label[idx]).long(),
        }


def mutate_labels(cls_label, p_mutate=0.1, temperature=3.0, num_heads=3, seed=0):
    """In-place mutation of class labels using temperature sampling.
    Args:
        cls_label: (B, N_heads) long tensor
    Returns:
        mutated cls_label (same tensor, modified in-place)
    """
    torch.manual_seed(seed)
    B, N = cls_label.shape
    # Per-head independent trigger
    for hi in range(N):
        mask = torch.rand(B) < p_mutate
        if not mask.any():
            continue
        # Simulate logits as uniform + small noise (in real impl, use saved elite logits)
        fake_logits = torch.randn(B, 24) * 2.0
        fake_logits[:, cls_label[0, hi].item()] += 5.0  # bias toward chosen class
        probs = F.softmax(fake_logits[mask] / temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        cls_label[mask, hi] = sampled
    return cls_label


def bc_mutate_one(model, loader, device, lr=1e-3, p_mutate=0.1, temperature=3.0, seed=0):
    """Lightweight mutation: 1 batch, only update policy_heads."""
    model.train()
    model.to(device)

    # Freeze everything except policy_heads
    for name, param in model.named_parameters():
        if not name.startswith("policy_heads"):
            param.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in model.policy_heads.parameters()], lr=lr
    )

    batch = next(iter(loader))
    board = batch["board"].to(device)
    stats = batch["stats"].to(device)
    cls_label = batch["class_label"].to(device).clone()

    # Mutate labels with this individual's seed
    mutate_labels(cls_label, p_mutate=p_mutate, temperature=temperature,
                  num_heads=model.num_heads, seed=seed)

    optimizer.zero_grad()
    output = model(board, stats)
    loss = 0.0
    for i in range(model.num_heads):
        loss += F.cross_entropy(output[f"head{i+1}_logits"], cls_label[:, i])
    loss /= model.num_heads
    loss.backward()
    optimizer.step()

    # Restore requires_grad
    for param in model.parameters():
        param.requires_grad = True

    model.cpu()
    model.eval()
    return model


def bench(num_individuals=100, batch_size=64):
    print(f"Benchmark: BC mutate {num_individuals} individuals")
    print(f"Batch size: {batch_size}")
    print("=" * 55)

    # ── Create synthetic model + data ────────────────────────────
    num_heads = 8
    print(f"\n1. Creating model ({num_heads} heads)...")
    model = create_model(num_heads=num_heads)
    # Random init
    for p in model.parameters():
        p.data.normal_(0, 0.02)
    model.eval()
    print(f"   model: {model.count_parameters():,} params")

    # Synthetic data: 5000 random samples
    print("\n2. Creating synthetic data (5000 samples)...")
    n_samples = 5000
    fake_board = np.random.randn(n_samples, 28, 19, 19).astype(np.float16)
    fake_stats = np.random.randn(n_samples, 42).astype(np.float16)
    fake_class = np.random.randint(0, 24, size=(n_samples, num_heads)).astype(np.int64)
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="bc_bench_"))
    from my_ai.elite_bc import write_bc_npz
    write_bc_npz(tmp_dir / "synthetic.npz", [fake_board], [fake_stats], [fake_class],
                  [np.zeros((n_samples, num_heads), dtype=np.int64)])

    ds = MutateDataset([tmp_dir / "synthetic.npz"], batch_size=batch_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    print(f"   dataset: {len(ds)} samples, {len(loader)} batches/epoch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   device: {device}")

    # ── Mutate N individuals ────────────────────────────────────
    print(f"\n3. Mutating {num_individuals} individuals...")
    params_list = []
    t0 = time.perf_counter()

    for i in range(num_individuals):
        # Deep copy model
        model_i = create_model(num_heads=num_heads)
        model_i.load_state_dict(model.state_dict())
        model_i.eval()

        # One batch, only update heads
        bc_mutate_one(model_i, loader, device, seed=i, p_mutate=0.1, temperature=3.0)

        # Add tiny sigma noise (0.0002)
        params_i = model_i.get_parameters_as_vector()
        noise = np.random.randn(*params_i.shape).astype(np.float32) * 0.0002
        params_list.append(params_i + noise)

        if (i + 1) % 20 == 0:
            elapsed = time.perf_counter() - t0
            print(f"   {i+1}/{num_individuals} done ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t0
    print(f"\n4. Results:")
    print(f"   Total: {total_time:.1f}s for {num_individuals} individuals")
    print(f"   Avg:   {total_time / num_individuals * 1000:.1f}ms per individual")

    # ── Check diversity ─────────────────────────────────────────
    print(f"\n5. Diversity check:")
    mean_vec = model.get_parameters_as_vector()
    diffs = [np.abs(p - mean_vec).mean() for p in params_list]
    print(f"   Mean param diff from base: {np.mean(diffs):.6f} ± {np.std(diffs):.6f}")
    print(f"   Max diff: {max(diffs):.6f}, Min diff: {min(diffs):.6f}")

    # Pairwise diversity
    if len(params_list) > 1:
        pair_diffs = []
        for i in range(min(10, len(params_list))):
            for j in range(i + 1, min(10, len(params_list))):
                pair_diffs.append(np.abs(params_list[i] - params_list[j]).mean())
        print(f"   Pairwise diff (first 10): {np.mean(pair_diffs):.6f} ± {np.std(pair_diffs):.6f}")

    print("\n" + "=" * 55)
    print("DONE" if total_time < 120 else "TOO SLOW (>2min)")
    print("=" * 55)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    bench()
