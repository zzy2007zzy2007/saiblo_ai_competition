#!/usr/bin/env python3
"""亚军 value 函数蒸馏：只训 value_state，回归 evaluate_terminal 标签（架构表达力验证）。

预注册：docs/prereg_20261019_distill_value_expressivity.md

用法（训练环境 python）:
  D:/anaconda3/envs/pytorch-gpu/python.exe -u code/my_ai/distill_top2/train_value_distill.py \
      --data training_history/distill_top2/value_ds_12g.npz \
      --ckpt checkpoints/grafted_sdk_A.pt --train-mode full \
      --out training_history/distill_top2/value_distill_gap.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code", _REPO / "code" / "cpp_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from my_ai.network import create_model, model_kwargs_from_ckpt  # noqa: E402


# ── 指标 ────────────────────────────────────────────────────────────────────
def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-8
    Xn = (X - mu) / sd
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb.T @ y)
    return mu, sd, w


def pred_ridge(mu, sd, w, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xn = (X - mu) / sd
    Xb = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    return Xb @ w


# ── 数据 ────────────────────────────────────────────────────────────────────
def parse_seeds(spec: str) -> set[int]:
    out: set[int] = set()
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
    ap.add_argument("--data", default="training_history/distill_top2/value_ds_12g.npz")
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--out", default="training_history/distill_top2/value_distill.pt")
    ap.add_argument("--value-pool", default="gap")
    ap.add_argument("--train-mode", choices=["head", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-seeds", default="108-111")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--zero-stats", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    d = np.load(args.data)
    board = d["board"].astype(np.float32)   # (N,28,19,19)
    stats = d["stats"].astype(np.float32)   # (N,42)
    value = d["value"].astype(np.float64)   # (N,)
    seeds = d["seed"]
    if args.zero_stats:
        stats = np.zeros_like(stats)

    val_seeds = parse_seeds(args.val_seeds)
    val_mask = np.isin(seeds, list(val_seeds))
    tr_mask = ~val_mask
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.where(val_mask)[0]
    print(f"[data] N={len(value)} train={len(tr_idx)} val={len(va_idx)} "
          f"train_seeds={sorted(set(seeds[tr_mask].tolist()))} val_seeds={sorted(val_seeds)}", flush=True)
    print(f"[data] value mean/std(min,max) train={value[tr_idx].mean():.1f}/{value[tr_idx].std():.1f} "
          f"val={value[va_idx].mean():.1f}/{value[va_idx].std():.1f}", flush=True)

    # 目标归一化（用 train 统计量）
    y_mean = float(value[tr_idx].mean())
    y_std = float(value[tr_idx].std()) + 1e-8
    y_norm = (value - y_mean) / y_std

    # ── 模型 ──
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    kw = model_kwargs_from_ckpt(ck)
    kw["value_pool"] = args.value_pool
    model = create_model(**kw)
    vs = ck.get("value_state") or ck.get("model_state")
    if vs is None:
        raise SystemExit("ckpt 里没有 value_state / model_state")
    model_sd = model.state_dict()
    load_sd, missing = {}, []
    for k, v in vs.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            load_sd[k] = v
        else:
            missing.append(k)
    model.load_state_dict(load_sd, strict=False)
    n_loaded = len(load_sd)
    print(f"[model] value_pool={args.value_pool} params={model.count_parameters():,} "
          f"loaded={n_loaded}/{len(model_sd)} 未加载={len(missing)}", flush=True)
    if missing:
        print(f"[model] 未加载(随机初始化): {missing[:8]}{' ...' if len(missing) > 8 else ''}", flush=True)

    model.to(args.device)
    if args.train_mode == "head":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("value_head.")
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] head-only: 可训参数={n_trainable:,}", flush=True)
    else:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[model] full: 可训参数={n_trainable:,}", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    loss_fn = nn.MSELoss()

    tr_board = torch.from_numpy(board[tr_idx])
    tr_stats = torch.from_numpy(stats[tr_idx])
    tr_y = torch.from_numpy(y_norm[tr_idx].astype(np.float32)).unsqueeze(1)
    va_board = torch.from_numpy(board[va_idx])
    va_stats = torch.from_numpy(stats[va_idx])

    best_r2 = -1e9
    best_epoch = -1
    best_sd = None
    no_improve = 0
    n_tr = len(tr_idx)

    def evaluate() -> tuple[float, np.ndarray]:
        model.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(va_idx), 512):
                b = va_board[start:start + 512].to(args.device)
                s = va_stats[start:start + 512].to(args.device)
                outs.append(model(b, s)["value"].squeeze(1).cpu().numpy())
        pred_norm = np.concatenate(outs)
        pred = pred_norm * y_std + y_mean
        return r2_score(value[va_idx], pred), pred

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_tr)
        total_loss = 0.0
        for start in range(0, n_tr, args.batch_size):
            idx = perm[start:start + args.batch_size]
            b = tr_board[idx].to(args.device)
            s = tr_stats[idx].to(args.device)
            y = tr_y[idx].to(args.device)
            pred = model(b, s)["value"]
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(idx) / n_tr
        r2, _ = evaluate()
        improved = ""
        if r2 > best_r2:
            best_r2 = r2
            best_epoch = ep
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            improved = "  *"
        else:
            no_improve += 1
        print(f"[ep {ep:3d}] train_mse={total_loss:.4f} val_R2={r2:.4f}{improved}", flush=True)
        if no_improve >= args.patience:
            print(f"[stop] early stop @ep {ep} (patience {args.patience})", flush=True)
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    r2_net, pred = evaluate()
    mae_net = float(np.mean(np.abs(value[va_idx] - pred)))
    print(f"\n[result] best_epoch={best_epoch} val_R2_net={r2_net:.4f} val_MAE_net={mae_net:.2f}",
          flush=True)

    # ── 线性基线（只用 stats）──
    for tag, X in (("stats42", stats), ("summary22", stats[:, :22])):
        mu, sd, w = fit_ridge(X[tr_idx], value[tr_idx])
        lin_pred = pred_ridge(mu, sd, w, X[va_idx])
        print(f"[baseline] linear_{tag:<10} val_R2={r2_score(value[va_idx], lin_pred):.4f} "
              f"val_MAE={float(np.mean(np.abs(value[va_idx] - lin_pred))):.2f}", flush=True)

    # 常数基线
    const_pred = np.full(len(va_idx), value[tr_idx].mean())
    print(f"[baseline] constant           val_R2={r2_score(value[va_idx], const_pred):.4f}", flush=True)

    # ── 保存 ──
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "value_state": model.state_dict(),
        "value_pool": args.value_pool,
        "train_mode": args.train_mode,
        "data": args.data,
        "val_seeds": sorted(val_seeds),
        "val_R2": r2_net,
        "val_MAE": mae_net,
        "y_mean": y_mean,
        "y_std": y_std,
        "num_resblocks": kw["num_resblocks"],
        "num_heads": kw["num_heads"],
        "latent_dim": kw["latent_dim"],
        "no_bn": kw["no_bn"],
        "gn": kw["gn"],
        "gn_groups": kw["gn_groups"],
    }, out)
    print(f"[save] {out}  ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
