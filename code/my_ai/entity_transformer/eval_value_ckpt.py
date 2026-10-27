#!/usr/bin/env python3
"""在**指定的一批样本**上评测价值网 checkpoint（用于跨配置可比）。

默认用 `lowq_f64` 的验证集（最后 500 局的 64 帧/局 = 32,000 条），
这样"64 帧版"和"全量帧版"的模型可以在**完全相同的样本**上比 R²。

用法：
  python code/my_ai/entity_transformer/eval_value_ckpt.py \
      --ckpt checkpoints/full/lowq_seed0_ep2_r0.2923.pt \
      --data D:/.../lowq_f64 --val-games 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "code"))
from my_ai.value_network import ValueNetwork  # noqa: E402


def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="含 board/stats + 标签 + 局信息的目录")
    ap.add_argument("--label-file", default="label.npy")
    ap.add_argument("--val-games", type=int, default=500)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    d = Path(a.data)
    if not d.is_absolute():
        d = _REPO / d
    ck = Path(a.ckpt)
    if not ck.is_absolute():
        ck = _REPO / ck

    board = np.load(d / "board.npy", mmap_mode="r")
    stats = np.load(d / "stats.npy", mmap_mode="r")
    y = np.load(d / a.label_file).astype(np.float64)

    if (d / "game_offsets.npy").exists():
        offs = np.load(d / "game_offsets.npy")
    else:
        gid = np.load(d / "game_id.npy")
        offs = np.searchsorted(gid, np.arange(int(gid[-1]) + 2))
    n_games = len(offs) - 1
    lo = int(offs[max(0, n_games - a.val_games)])
    hi = int(offs[-1])
    idx = np.arange(lo, hi)

    c = torch.load(ck, map_location="cpu", weights_only=False)
    # ⚠️ 按 ckpt 里的 `config` 建网：32/3 的小网不能再用写死的 `ValueNetwork()`。
    # 老 ckpt 的 config 就是 64/6 ⇒ 不传参数时行为与以前完全一致。
    _cfg = c.get("config") or {}
    model = ValueNetwork(latent_dim=int(_cfg.get("latent_dim", 64)),
                         num_resblocks=int(_cfg.get("num_resblocks", 6)))
    model.load_state_dict(c["state_dict"])
    model.eval()
    ym, ys = float(c["y_mean"]), float(c["y_std"])

    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), 1024):
            bi = idx[s:s + 1024]
            b = torch.from_numpy(np.array(board[bi], np.float32))
            st = torch.from_numpy(np.array(stats[bi], np.float32))
            preds.append(model(b, st).reshape(-1).numpy())
    p = np.concatenate(preds) * ys + ym
    t = y[idx]
    print(f"[eval] {ck.name}")
    print(f"  data={d.name} label={a.label_file} val_rows={len(idx)} (games {n_games-a.val_games}..{n_games-1})")
    print(f"  R2={r2(t, p):.4f}  MAE={np.mean(np.abs(t-p)):.4f}  label_std={t.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
