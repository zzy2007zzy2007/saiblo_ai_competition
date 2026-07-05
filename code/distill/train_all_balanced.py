"""Train small model on ALL data (incl HOLD) with per-class balanced sampling.
This gives the encoder maximum diversity (133K samples) while ensuring
each class gets ~equal representation in every batch."""
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


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_paths = sorted(Path("distill_data_v2").glob("distill_seed*.npz"))
    orig = [p for p in all_paths if int(p.stem.replace("distill_seed", "")) < 500]
    print(f"Files: {len(orig)}")

    # Use ALL data (p_hold=1.0 = keep all HOLD)
    ds = SSDataset(orig, p_hold=1.0)
    print(f"Samples: {len(ds)}")

    # Per-class balanced sampling weights
    labels = ds.class_label[:, 0]  # (T,)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Classes: {len(unique)}")
    for c, cnt in sorted(zip(unique.tolist(), counts.tolist()),
                          key=lambda x: -x[1]):
        print(f"  class {c}: {cnt} ({100*cnt/len(labels):.1f}%)")

    # Sample weight = 1/count for each class (balanced sampling)
    class_w = {c: 1.0 / cnt for c, cnt in zip(unique, counts)}
    sample_weights = np.array([class_w[l] for l in labels])
    sample_weights /= sample_weights.mean()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights), replacement=True)

    model = create_model(small=True)
    print(f"Model: {model.count_parameters():,} params")
    model.to(device)

    # CE class weights (not needed with balanced sampling, but helpful)
    # Per-sample weight = 1/count means each class gets equal total weight
    cw = None  # no need for additional CE class weights

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_epochs = 300
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    model.train()

    for epoch in range(total_epochs):
        loader = DataLoader(ds, batch_size=64, sampler=sampler)
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            board = batch["board"].to(device)
            stats = batch["stats"].to(device)
            cls_label = batch["class_label"].to(device)
            target_map = batch["action_map"].to(device)

            # Augmentation: LR flip + noise + cutout
            if torch.rand(1).item() < 0.5:
                board = board.flip(-1)
                target_map = target_map.flip(-1)
            # Gaussian noise to board (small, since board is ~0-1)
            board = board + torch.randn_like(board) * 0.02
            # Cutout: mask random 4x4 patches
            if torch.rand(1).item() < 0.3:
                bs, ch, h, w = board.shape
                cut_h, cut_w = 4, 4
                y = torch.randint(0, h - cut_h, (bs,))
                x = torch.randint(0, w - cut_w, (bs,))
                for b in range(bs):
                    board[b, :, y[b]:y[b]+cut_h, x[b]:x[b]+cut_w] = 0.0

            optimizer.zero_grad()
            output = model(board, stats)

            cls_loss = _ce_with_label_smoothing(
                output["head1_logits"], cls_label[:, 0],
                smoothing=0.1, weight=cw)

            B, C, H, W = target_map.shape
            pred_flat = output["action_map"].view(B, C, -1).permute(0, 2, 1)
            target_flat = target_map.view(B, C, -1).permute(0, 2, 1)
            map_loss = F.kl_div(
                F.log_softmax(pred_flat, dim=-1),
                F.softmax(target_flat.detach(), dim=-1),
                reduction="batchmean")

            soft_loss = torch.tensor(0.0, device=device)
            if "class_scores" in batch:
                soft_target = batch["class_scores"].to(device)
                soft_dist = F.softmax(soft_target / 2.0, dim=-1).detach()
                soft_loss = F.kl_div(
                    F.log_softmax(output["head1_logits"] / 2.0, dim=-1),
                    soft_dist, reduction="batchmean")

            bias_reg = model.policy_heads[0].bias.pow(2).sum()
            loss = cls_loss + map_loss + 0.5 * soft_loss + 0.01 * bias_reg
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch+1}/{total_epochs}: cls={cls_loss.item():.4f} "
                  f"tot={total_loss/n_batches:.4f} lr={lr_now:.6f}")

    model.cpu().eval()
    mean = model.get_parameters_as_vector()
    save_checkpoint(Path("distill_data_v2/model_all_bal300.pt"), mean, model, 0)
    print("Saved: model_all_bal300.pt")

    # Quick test
    board = torch.randn(20, 28, 19, 19)
    st = torch.randn(20, 42)
    with torch.no_grad():
        preds = model(board, st)["head1_logits"].argmax(axis=1).numpy()
        print(f"Random test: {len(np.unique(preds))} unique: {np.unique(preds)}")


if __name__ == "__main__":
    train()
