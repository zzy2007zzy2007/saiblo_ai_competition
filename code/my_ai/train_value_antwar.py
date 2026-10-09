"""训练**部署用**的价值网（`AntWarNetwork` 形状的 `value_state`），并直接产出可部署的三网 ckpt。

为什么需要这个脚本（而不是 `train_value_net.py`）：
- 部署端 `load_three_models` / `make_three_net_fn` 要求 `value_state` 是**完整 `AntWarNetwork`** 的 state_dict（94 张量），
  而 `train_value_net.py` 产出的是**独立小网 `ValueNetwork`**（542,913 参数）⇒ 键名交集 0、无法嫁接（2026-10-09 实测）。
- 本脚本：`create_model(**model_kwargs_from_ckpt(base))` 造同架构网 → 用 `MSE(tanh(value) , target)` 训练
  （部署里 `value = tanh(裸输出)`，见 `az_selfplay.make_three_net_fn`）→ 把训练好的 state_dict **只替换 `value_state`**。
- **按局切分**训练/验证（每个 npz 文件 = 一局一方 ⇒ 按文件切），避免同局相邻帧泄漏。

数据：`pkl_to_value_npz.py` 的产物（`board (T,28,19,19) f16` / `stats (T,42) f16` / `target (T,) f32`，target = 该样本视角的终局价值）。

用法：
    python code/my_ai/train_value_antwar.py --data-dirs data/s2_value_npz \\
        --base checkpoints/grafted_sdk_A.pt --out checkpoints/grafted_sdk_A_vs2.pt \\
        --epochs 30 --batch-size 256 --lr 1e-3 --val-games 12
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from my_ai.network import create_model, model_kwargs_from_ckpt  # noqa: E402


def load_games(dirs: list[Path]) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    games = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f)
            games.append((z["board"], z["stats"], z["target"].astype(np.float32)))
    return games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+", required=True)
    ap.add_argument("--base", required=True, help="部署三网 ckpt（提供 class/pos/元数据，且决定架构）")
    ap.add_argument("--out", required=True, help="输出三网 ckpt（class/pos 原样、value_state 换成新网）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-games", type=int, default=12, help="按局切分：留出多少局做验证")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-net", default="", help="可选：另外单独保存训练好的网（不含 class/pos）")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base = torch.load(a.base, map_location="cpu", weights_only=False)
    kw = model_kwargs_from_ckpt(base)
    model = create_model(**kw).to(dev)
    cur = model.state_dict()
    want = base["value_state"]
    miss = [k for k in cur if k not in want]; bad = [k for k in cur if k in want and tuple(cur[k].shape) != tuple(want[k].shape)]
    print(f"   架构核对：本网 {len(cur)} 张量 vs 部署 value_state {len(want)} 张量；缺 {len(miss)}、形状不符 {len(bad)}")
    if miss or bad:
        print("   ❌ 架构不匹配 ⇒ 停（避免产出不可加载的 ckpt）"); return 2

    games = load_games([Path(d) for d in a.data_dirs])
    rng = np.random.default_rng(a.seed); idx = rng.permutation(len(games))
    n_val = max(1, min(a.val_games, len(games) // 5))
    val_g = [games[i] for i in idx[:n_val]]; tr_g = [games[i] for i in idx[n_val:]]
    n_tr = sum(g[0].shape[0] for g in tr_g); n_va = sum(g[0].shape[0] for g in val_g)
    print(f"   数据：训练 {len(tr_g)} 局/{n_tr} 帧，验证 {len(val_g)} 局/{n_va} 帧（**按局切分**）")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    best = (1e9, None)
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        order = rng.permutation(len(tr_g))
        batches = []
        for gi in order:
            b, s, t = tr_g[gi]
            for i in range(0, b.shape[0], a.batch_size):
                batches.append((b[i:i + a.batch_size], s[i:i + a.batch_size], t[i:i + a.batch_size]))
        rng.shuffle(batches)
        for bb, ss, tt in batches:
            xb = torch.from_numpy(bb.astype(np.float32)).to(dev)
            xs = torch.from_numpy(ss.astype(np.float32)).to(dev)
            yt = torch.from_numpy(tt).to(dev)
            v = torch.tanh(model(xb, xs)["value"].squeeze(-1))
            loss = lossf(v, yt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        model.eval(); va = []
        with torch.no_grad():
            for b, s, t in val_g:
                xb = torch.from_numpy(b.astype(np.float32)).to(dev); xs = torch.from_numpy(s.astype(np.float32)).to(dev)
                v = torch.tanh(model(xb, xs)["value"].squeeze(-1)).cpu().numpy()
                va.append(np.abs(v - t))
        mae = float(np.concatenate(va).mean())
        flag = ""
        if mae < best[0]:
            best = (mae, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}); flag = "  ← best"
        print(f"   [{time.strftime('%H:%M:%S')}] epoch {ep:3d}/{a.epochs}  train_mse={tot / max(nb,1):.5f}  val_mae={mae:.4f}  ({time.time() - t0:.1f}s){flag}", flush=True)

    assert best[1] is not None
    out = dict(base)
    out["value_state"] = best[1]
    torch.save(out, a.out)
    same_cs = all(torch.equal(out["class_state"][k], base["class_state"][k]) for k in base["class_state"])
    same_ps = all(torch.equal(out["pos_state"][k], base["pos_state"][k]) for k in base["pos_state"])
    print(f"   ✅ 已写出 {a.out}（best val_mae={best[0]:.4f}）")
    print(f"   class_state 与原文件逐位相同={same_cs}；pos_state 逐位相同={same_ps}；value_state 已替换（{len(best[1])} 张量）")
    if a.save_net:
        torch.save({"model_state": best[1], "val_mae": best[0]}, a.save_net)
        print(f"   （另存训练网 -> {a.save_net}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
