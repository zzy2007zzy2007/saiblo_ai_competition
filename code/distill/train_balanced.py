"""Train small model with balanced sampling + augmentation on non-HOLD data."""
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


def train(data_dir="distill_data_v2", out="distill_data_v2/model_aug.pt",
          epochs=200, lr=1e-3, batch_size=64,
          label_smoothing=0.1, weight_decay=1e-4, bias_decay=1e-2,
          lambda_soft=0.5, soft_temp=2.0, max_seed=None):
    # Load files (skip corrupt), optionally filter by max seed
    all_paths = sorted(Path(data_dir).glob("distill_seed*.npz"))
    if max_seed is not None:
        all_paths = [p for p in all_paths
                     if int(p.stem.replace("distill_seed", "")) < max_seed]
        print(f"  Filtered seed < {max_seed}: {len(all_paths)} files")
    valid_paths = all_paths  # skip validation; all < 500 seed files are clean
    print(f"Files: {len(valid_paths)}")

    ds = SSDataset(valid_paths, p_hold=0.0)
    print(f"Samples: {len(ds)}")

    # Balanced sampling
    labels = ds.class_label[:, 0]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Classes: {len(unique)}  distribution: {dict(zip(unique.tolist(), counts.tolist()))}")
    class_weight_map = {c: 1.0 / cnt for c, cnt in zip(unique, counts)}
    sample_weights = np.array([class_weight_map[l] for l in labels])
    sample_weights /= sample_weights.mean()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights), replacement=True)

    # Model
    model = create_model(small=True)
    print(f"Model: {model.count_parameters():,} params, small=True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    # Class weights for CE
    cw_tensor = torch.zeros(24)
    for c, w in class_weight_map.items():
        cw_tensor[int(c)] = w
    cw_tensor = cw_tensor / cw_tensor.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()

    for epoch in range(epochs):
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
        total_loss = epoch_cls = epoch_map = 0.0
        n_batches = 0
        for batch in loader:
            board = batch["board"].to(device)
            stats = batch["stats"].to(device)
            cls_label = batch["class_label"].to(device)
            target_map = batch["action_map"].to(device)

            # Augmentation: random LR flip
            if torch.rand(1).item() < 0.5:
                board = board.flip(-1)
                target_map = target_map.flip(-1)

            optimizer.zero_grad()
            output = model(board, stats)

            cls_loss = _ce_with_label_smoothing(
                output["head1_logits"], cls_label[:, 0],
                smoothing=label_smoothing, weight=cw_tensor.to(device))

            B, C, H, W = target_map.shape
            pred_flat = output["action_map"].view(B, C, -1).permute(0, 2, 1)
            target_flat = target_map.view(B, C, -1).permute(0, 2, 1)
            map_loss = F.kl_div(
                F.log_softmax(pred_flat, dim=-1),
                F.softmax(target_flat.detach(), dim=-1),
                reduction="batchmean")

            soft_loss = torch.tensor(0.0, device=device)
            if lambda_soft > 0 and "class_scores" in batch:
                soft_target = batch["class_scores"].to(device)
                soft_dist = F.softmax(soft_target / soft_temp, dim=-1).detach()
                soft_loss = F.kl_div(
                    F.log_softmax(output["head1_logits"] / soft_temp, dim=-1),
                    soft_dist, reduction="batchmean")

            bias_reg = model.policy_heads[0].bias.pow(2).sum()
            loss = cls_loss + map_loss + lambda_soft * soft_loss + bias_decay * bias_reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_cls += cls_loss.item()
            epoch_map += map_loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch+1}/{epochs}: cls={epoch_cls/n_batches:.4f} "
                  f"map={epoch_map/n_batches:.4f} tot={total_loss/n_batches:.4f} lr={lr_now:.6f}")

    model.cpu().eval()
    mean = model.get_parameters_as_vector()
    save_checkpoint(Path(out), mean, model, 0)
    print(f"Saved: {out}")

    # Quick random test
    board = torch.randn(20, 28, 19, 19)
    stats_ = torch.randn(20, 42)
    with torch.no_grad():
        logits = model(board, stats_)["head1_logits"].numpy()
        preds = logits.argmax(axis=1)
        print(f"Random test: {len(np.unique(preds))} unique classes: {np.unique(preds)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="distill_data_v2")
    parser.add_argument("--out", default="distill_data_v2/model_aug.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bias-decay", type=float, default=1e-2)
    parser.add_argument("--lambda-soft", type=float, default=0.5)
    parser.add_argument("--soft-temp", type=float, default=2.0)
    parser.add_argument("--max-seed", type=int, default=None,
                        help="only use files with seed < max_seed")
    args = parser.parse_args()
    train(**vars(args))
