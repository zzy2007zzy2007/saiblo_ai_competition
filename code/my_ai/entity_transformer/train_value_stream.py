#!/usr/bin/env python3
"""流式训练价值网（吃 `merge_lowq_data.py` 的大 memmap 数组）。

要点：
  * **按局切验证集**（用 `game_offsets.npy`）——同局样本高度相关，按帧切会泄漏；
  * 标签可选 `--label hp`（`(hp_us-hp_opp)/HP_SCALE`，历史口径）或 `win`（±0.5）；
  * **多个种子**（`--seeds 0 1 2`）——历史教训：单种子在 ±20pp 上无效；
  * 网络用 `ValueNetwork`（与 `AntWarNetwork` 的 value 通路**逐层一致**，权重可映射进 M6c 的 `value_state`）。

用法：
  python code/my_ai/entity_transformer/train_value_stream.py \
      --data "F:/2026智能体大赛_新2/training_history/lowq_e2_merged" \
      --label hp --seeds 0 1 2 --epochs 10 --batch-size 256
"""
from __future__ import annotations

import argparse
import json
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

from my_ai.value_network import ValueNetwork  # noqa: E402

HP_SCALE = 20.0


def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return 1.0 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="hp", choices=["hp", "win"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-games", type=int, default=500)
    ap.add_argument("--shard-games", type=int, default=5,
                    help="每片包含多少局；片是 I/O 与打乱的基本单位")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    d = Path(a.data)
    if not d.is_absolute():
        d = _REPO / d
    board = np.load(d / "board.npy", mmap_mode="r")
    stats = np.load(d / "stats.npy", mmap_mode="r")
    hp_us = np.load(d / "hp_us.npy")
    hp_opp = np.load(d / "hp_opp.npy")
    vwin = np.load(d / "value_win.npy")
    offs = np.load(d / "game_offsets.npy")
    N = board.shape[0]
    n_games = len(offs) - 1
    print(f"[data] N={N} games={n_games} label={a.label}", flush=True)

    # 按局切：最后 val_games 局做验证
    val_start_game = max(0, n_games - a.val_games)
    val_lo, val_hi = int(offs[val_start_game]), int(offs[-1])
    tr_mask = np.ones(N, dtype=bool)
    tr_mask[val_lo:val_hi] = False
    tr_idx = np.where(tr_mask)[0]
    va_idx = np.arange(val_lo, val_hi)
    print(f"[data] train={len(tr_idx)} val={len(va_idx)} (val games {val_start_game}..{n_games-1})",
          flush=True)

    # ⚠️ value_win.npy 就是**局末**血量差/20（az_selfplay 在局末算的，平局且分胜负给 ±0.1）。
    # **不要**用 hp_us/hp_opp 现算——那是**逐回合**的当前血量差，不是标签。
    y = np.asarray(vwin, np.float32)
    print(f"[data] label=terminal hp-diff/20 (value_win.npy) range=[{y.min():.2f},{y.max():.2f}]",
          flush=True)
    y_mean, y_std = float(y[tr_idx].mean()), float(y[tr_idx].std()) + 1e-8
    print(f"[data] label mean/std = {y_mean:.3f}/{y_std:.3f}", flush=True)

    # ── 分片：每 shard_games 局一片；片是 I/O 与打乱的基本单位 ──────────────
    # 片内样本在磁盘上连续（顺序 I/O），一个 epoch 内**打乱片序** + **片内打乱**，
    # 这样 batch 会混合多局（而不是同一局的相邻帧），同时避免 86 GB 上的随机散读。
    tr_games = np.arange(0, val_start_game)
    shard_lo = 0
    shards = []
    while shard_lo < len(tr_games):
        shard_hi = min(shard_lo + a.shard_games, len(tr_games))
        s = int(offs[tr_games[shard_lo]])
        e = int(offs[tr_games[shard_hi - 1] + 1])
        shards.append((s, e, shard_hi - shard_lo))
        shard_lo = shard_hi
    print(f"[data] {len(shards)} shards x ~{a.shard_games} games "
          f"(~{len(tr_idx)//max(len(shards),1)} samples/shard)", flush=True)

    dev = a.device

    def batch_tensor(idx):
        b = torch.from_numpy(np.asarray(board[idx], np.float32)).to(dev)
        s = torch.from_numpy(np.asarray(stats[idx], np.float32)).to(dev)
        t = torch.from_numpy(((y[idx] - y_mean) / y_std).astype(np.float32)).unsqueeze(1).to(dev)
        return b, s, t

    # ⚠️ 不要预载验证集：500 局 ≈ 43 万样本，float32 的 board 要 ~17 GB。
    # 改成每 512 条从 memmap 分块读（慢一点但内存安全）。
    y_va = y[va_idx]

    outdir = Path(a.out_dir)
    if not outdir.is_absolute():
        outdir = _REPO / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    for seed in a.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        model = ValueNetwork().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        best = -1e9; best_sd = None; best_ep = -1
        t0 = time.time()
        for ep in range(1, a.epochs + 1):
            model.train()
            tl, nseen = 0.0, 0
            order = np.random.permutation(len(shards))   # ① 打乱片序
            for si in order:
                s0, e0, _ng = shards[int(si)]
                # ① 整片顺序读入内存（保持 f16，省一半内存）——避免在 86 GB 文件上散读
                b_mem = np.asarray(board[s0:e0])
                s_mem = np.asarray(stats[s0:e0])
                y_mem = y[s0:e0]
                # ② 内存内打乱（片已含多局 ⇒ batch 会混合多局，而不是同局相邻帧）
                perm2 = np.random.permutation(len(b_mem))
                for k in range(0, len(perm2), a.batch_size):
                    bi = perm2[k:k + a.batch_size]
                    b = torch.from_numpy(b_mem[bi].astype(np.float32)).to(dev)
                    st = torch.from_numpy(s_mem[bi].astype(np.float32)).to(dev)
                    t = torch.from_numpy(((y_mem[bi] - y_mean) / y_std).astype(np.float32)
                                         ).unsqueeze(1).to(dev)
                    pred = model(b, st)
                    loss = loss_fn(pred, t)
                    opt.zero_grad(); loss.backward(); opt.step()
                    tl += float(loss.item()) * len(bi)
                    nseen += len(bi)
                del b_mem, s_mem, y_mem
            tl /= max(nseen, 1)
            model.eval()
            outs = []
            with torch.no_grad():
                for s in range(0, len(va_idx), 512):
                    chunk = va_idx[s:s+512]
                    vb = torch.from_numpy(np.asarray(board[chunk], np.float32)).to(dev)
                    vs = torch.from_numpy(np.asarray(stats[chunk], np.float32)).to(dev)
                    outs.append(model(vb, vs).squeeze(1).cpu().numpy())
            pv = np.concatenate(outs) * y_std + y_mean
            r = r2(y_va, pv); mae = float(np.mean(np.abs(y_va - pv)))
            star = ""
            if r > best:
                best, best_ep, star = r, ep, "  *"
                best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            print(f"  [seed {seed} ep {ep:2d}] train_mse={tl:.4f} val_R2={r:.4f} "
                  f"val_MAE={mae:.4f}{star} ({time.time()-t0:.0f}s)", flush=True)
        p = outdir / f"lowq_value_seed{seed}.pt"
        torch.save({"state_dict": best_sd, "config": {"latent_dim": 64, "num_resblocks": 6},
                    "label": a.label, "y_mean": y_mean, "y_std": y_std,
                    "val_r2": best, "best_ep": best_ep, "seed": seed,
                    "data": str(d)}, p)
        print(f"[seed {seed}] best val_R2={best:.4f} (ep {best_ep}) -> {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
