"""Train LARGE model (553K params, 3 heads) on ALL data with balanced sampling.

This is the large-model version of train_all_balanced.py. The key differences:
  - Uses create_model(small=False): 6 ResBlocks, 3 heads, 64 latent dim, 553K params
  - Averages CE loss across all 3 heads
  - Adds head diversity loss to prevent 3-head collapse (teacher is deterministic)
  - Same balanced sampling + augmentation as the small model recipe
"""
from __future__ import annotations
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from my_ai.network import create_model
from my_ai.ss_train import SSDataset, _ce_with_label_smoothing, save_checkpoint


def head_diversity_loss(head_logits: list[torch.Tensor]) -> torch.Tensor:
    """Encourage different heads to produce different distributions.

    Computes average pairwise symmetric KL divergence between heads.
    Returns a NEGATIVE loss (higher diversity = lower total loss).
    """
    n = len(head_logits)
    if n < 2:
        return torch.tensor(0.0, device=head_logits[0].device)
    div = 0.0
    count = 0
    for i in range(n):
        pi = F.softmax(head_logits[i], dim=-1)  # (B, 24)
        for j in range(i + 1, n):
            pj = F.softmax(head_logits[j], dim=-1)
            # Symmetric KL: KL(pi||pj) + KL(pj||pi)
            kl_ij = (pi * (pi.log() - pj.log())).sum(dim=-1).mean()
            kl_ji = (pj * (pj.log() - pi.log())).sum(dim=-1).mean()
            div += (kl_ij + kl_ji) / 2.0
            count += 1
    # Negative: high divergence → lower loss
    return -div / count


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    all_paths = sorted(Path("distill_data_v2").glob("distill_seed*.npz"))
    # Only use clean files (seed < 500)
    orig = [p for p in all_paths if int(p.stem.replace("distill_seed", "")) < 500]
    print(f"Files: {len(orig)}")

    # Use ALL data with balanced sampling
    ds = SSDataset(orig, p_hold=1.0)
    print(f"Samples: {len(ds)}")

    labels = ds.class_label[:, 0]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Classes: {len(unique)} out of 24")
    for c, cnt in sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: -x[1]):
        print(f"  class {c:2d}: {cnt:6d} ({100*cnt/len(labels):.1f}%)")

    # Balanced sampling weights
    class_w = {c: 1.0 / cnt for c, cnt in zip(unique, counts)}
    sample_weights = np.array([class_w[l] for l in labels])
    sample_weights /= sample_weights.mean()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights), replacement=True)

    # --- Model (LARGE: 553K params, 3 heads) ---
    model = create_model(small=False)  # 6 ResBlocks, 64 latent, 3 heads
    print(f"Model: {model.count_parameters():,} params, "
          f"{model.num_resblocks} ResBlocks, {model.num_heads} heads, "
          f"latent_dim={model.LATENT_DIM}")
    model.to(device)

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_epochs = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    model.train()

    lambda_div = 0.1     # head diversity loss weight
    lambda_map = 0.3     # action map KL weight (reduced from 1.0 to prevent logit explosion)
    lambda_bias = 0.01   # bias regularizer

    for epoch in range(total_epochs):
        loader = DataLoader(ds, batch_size=64, sampler=sampler)
        total_loss = 0.0
        total_cls = 0.0
        total_map = 0.0
        total_dv = 0.0
        n_batches = 0

        for batch in loader:
            board = batch["board"].to(device)
            stats = batch["stats"].to(device)
            cls_label = batch["class_label"].to(device)  # (B, num_heads) or (B,)
            target_map = batch["action_map"].to(device)

            # Augmentation: LR flip
            if torch.rand(1).item() < 0.5:
                board = board.flip(-1)
                target_map = target_map.flip(-1)
            # Gaussian noise
            board = board + torch.randn_like(board) * 0.02
            # Cutout
            if torch.rand(1).item() < 0.3:
                bs, ch, h, w = board.shape
                cut_h, cut_w = 4, 4
                y = torch.randint(0, h - cut_h, (bs,))
                x = torch.randint(0, w - cut_w, (bs,))
                for b in range(bs):
                    board[b, :, y[b]:y[b]+cut_h, x[b]:x[b]+cut_w] = 0.0

            optimizer.zero_grad()
            output = model(board, stats)

            # --- Losses ---
            # 1) Classification loss: average over all 3 heads
            cls_loss = 0.0
            head_logits_list = []
            for i in range(model.num_heads):
                key = f"head{i+1}_logits"
                logits = output[key]
                head_logits_list.append(logits)
                # All heads share the same class label (teacher is deterministic)
                label = cls_label[:, 0] if cls_label.dim() == 2 else cls_label
                cls_loss += _ce_with_label_smoothing(
                    logits, label, smoothing=0.1, weight=None)
            cls_loss = cls_loss / model.num_heads

            # 2) Action map KL loss
            B, C, H, W = target_map.shape
            pred_flat = output["action_map"].view(B, C, -1).permute(0, 2, 1)  # (B, 361, 24)
            target_flat = target_map.view(B, C, -1).permute(0, 2, 1)  # (B, 361, 24)
            map_loss = F.kl_div(
                F.log_softmax(pred_flat, dim=-1),
                F.softmax(target_flat.detach(), dim=-1),
                reduction="batchmean")

            # 3) Head diversity loss
            div_loss = head_diversity_loss(head_logits_list)

            # 4) Bias regularization (prevent heads from relying on bias shortcuts)
            bias_reg = sum(h.bias.pow(2).sum() for h in model.policy_heads)

            # Combined loss
            loss = cls_loss + lambda_map * map_loss + lambda_div * div_loss + lambda_bias * bias_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_map += map_loss.item()
            total_dv += div_loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch+1}/{total_epochs}: cls={total_cls/n_batches:.4f} "
                  f"map={total_map/n_batches:.4f} div={total_dv/n_batches:.4f} "
                  f"tot={total_loss/n_batches:.4f} lr={lr_now:.6f}", flush=True)
        if (epoch + 1) % 50 == 0:
            # Save intermediate checkpoint
            model.cpu().eval()
            mean = model.get_parameters_as_vector()
            save_checkpoint(
                Path(f"distill_data_v2/model_large_bal_ep{epoch+1}.pt"),
                mean, model, 0)
            model.to(device).train()
            print(f"  Saved intermediate checkpoint at epoch {epoch+1}", flush=True)

    # --- Save ---
    model.cpu().eval()
    mean = model.get_parameters_as_vector()
    save_checkpoint(Path("distill_data_v2/model_large_bal.pt"), mean, model, 0)
    print("Saved: distill_data_v2/model_large_bal.pt")

    # --- Quick random test ---
    board = torch.randn(20, 28, 19, 19)
    st = torch.randn(20, 42)
    with torch.no_grad():
        for i in range(model.num_heads):
            key = f"head{i+1}_logits"
            preds = model(board, st)[key].argmax(axis=1).numpy()
            print(f"  {key}: {len(np.unique(preds))} unique: {np.unique(preds)}")


if __name__ == "__main__":
    train()
