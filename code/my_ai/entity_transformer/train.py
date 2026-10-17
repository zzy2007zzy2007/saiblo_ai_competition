#!/usr/bin/env python3
"""训练实体 token Transformer 的价值（第 1 步验证：能否超过 CNN 的 0.869 / 线性的 0.932）。

用法（训练环境 python）:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/entity_transformer/train.py \
      --data training_history/distill_top2/entity_ds.npz --val-seeds 914-917
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from my_ai.entity_transformer.model import EntityTransformerValue, count_params  # noqa: E402


def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def fit_ridge(X, y, lam=1.0):
    X = np.asarray(X, np.float64); y = np.asarray(y, np.float64)
    mu = X.mean(0); sd = X.std(0) + 1e-8
    Xn = (X - mu) / sd
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    w = np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
    return mu, sd, w


def pred_ridge(mu, sd, w, X):
    X = np.asarray(X, np.float64)
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1) @ w


def parse_seeds(spec):
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training_history/distill_top2/entity_ds.npz")
    ap.add_argument("--val-seeds", default="914-917")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    t0 = time.time()
    d = np.load(a.data)
    val_seeds = parse_seeds(a.val_seeds)
    seeds = d["seed"]
    vm = np.isin(seeds, list(val_seeds))
    tr = np.where(~vm)[0]; va = np.where(vm)[0]
    print(f"[data] N={len(seeds)} train={len(tr)} val={len(va)} "
          f"val_seeds={sorted(val_seeds)}", flush=True)

    value = d["value"].astype(np.float64)
    y_mean = float(value[tr].mean()); y_std = float(value[tr].std()) + 1e-8
    y_norm = ((value - y_mean) / y_std).astype(np.float32)

    dev = a.device
    def to_t(idx):
        return {
            "tower_cont": torch.from_numpy(d["tower_cont"][idx]).float().to(dev),
            "tower_type": torch.from_numpy(d["tower_type"][idx]).long().to(dev),
            "tower_pos": torch.from_numpy(d["tower_pos"][idx]).long().to(dev),
            "tower_mask": torch.from_numpy(d["tower_mask"][idx]).bool().to(dev),
            "ant_cont": torch.from_numpy(d["ant_cont"][idx]).float().to(dev),
            "ant_kind": torch.from_numpy(d["ant_kind"][idx]).long().to(dev),
            "ant_behavior": torch.from_numpy(d["ant_behavior"][idx]).long().to(dev),
            "ant_pos": torch.from_numpy(d["ant_pos"][idx]).long().to(dev),
            "ant_mask": torch.from_numpy(d["ant_mask"][idx]).bool().to(dev),
            "stats": torch.from_numpy(d["stats"][idx]).float().to(dev),
        }
    def to_t_batch(idx):
        b = to_t(idx)
        return b, torch.from_numpy(y_norm[idx]).unsqueeze(1).to(dev)

    model = EntityTransformerValue(d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads).to(dev)
    print(f"[model] params={count_params(model):,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    va_b = to_t(va)
    def evaluate():
        model.eval()
        outs = []
        with torch.no_grad():
            for s in range(0, len(va), 256):
                sl = slice(s, s + 256)
                bb = {k: v[sl] for k, v in va_b.items()}
                outs.append(model(bb).squeeze(1).cpu().numpy())
        pred = np.concatenate(outs) * y_std + y_mean
        return r2(value[va], pred), float(np.mean(np.abs(value[va] - pred)))

    best_r2, best_ep, no_imp, best_sd = -1e9, -1, 0, None
    n = len(tr)
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        tl = 0.0
        for s in range(0, n, a.batch_size):
            idx = tr[perm[s:s + a.batch_size]]
            bb, yy = to_t_batch(idx)
            pred = model(bb)
            loss = loss_fn(pred, yy)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += float(loss.item()) * len(idx) / n
        r, mae = evaluate()
        star = ""
        if r > best_r2:
            best_r2, best_ep, no_imp, star = r, ep, 0, "  *"
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
        print(f"[ep {ep:3d}] train_mse={tl:.4f} (train_R2={1-tl:.4f}) val_R2={r:.4f} val_MAE={mae:.1f}{star}", flush=True)
        if no_imp >= a.patience:
            print(f"[stop] early stop @ep {ep}", flush=True); break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    r, mae = evaluate()
    print(f"\n[result] best_ep={best_ep} val_R2={r:.4f} val_MAE={mae:.1f}", flush=True)

    # 线性基线（42 stats）
    stats = d["stats"].astype(np.float64)
    mu, sd, w = fit_ridge(stats[tr], value[tr])
    lp = pred_ridge(mu, sd, w, stats[va])
    print(f"[baseline] linear_stats42 val_R2={r2(value[va], lp):.4f} "
          f"val_MAE={float(np.mean(np.abs(value[va]-lp))):.1f}", flush=True)
    print(f"[done] {time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
