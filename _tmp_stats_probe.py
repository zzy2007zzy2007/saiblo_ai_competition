"""How much of a value head's output is explainable from the 42-dim stats ALONE?

Mechanistic diagnostic (no games needed).  For each checkpoint we
  1. forward N held-out states through its value net -> V
  2. fit a small MLP (stats(42) -> V) on 80% and report R^2 on 20%
  3. report corr(V, current hp_delta) -- the "copy stats[1]" shortcut indicator

Reading:
  high stats-R^2 (~1)  -> V is essentially a function of the stats (board-blind)
  lower stats-R^2      -> the board contributes to V
  corr(V, hp_delta)    -> how much V just echoes the current HP difference

Usage: python _tmp_stats_probe.py <ckpt> [<ckpt> ...]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent
sys.path[:0] = [str(REPO / "Ant-Game"), str(REPO / "code")]

from my_ai.network import create_model, model_kwargs_from_ckpt  # noqa: E402
from my_ai.az_intent.mcts import HP_SCALE  # noqa: E402


def build_value_net(ckpt: dict):
    """Return a net whose forward value output we should probe."""
    if "value_state" in ckpt:
        kw = dict(model_kwargs_from_ckpt(ckpt))
        kw["no_bn"] = bool(ckpt.get("value_no_bn", False))
        kw["gn"] = bool(ckpt.get("value_gn", False))
        kw["gn_groups"] = int(ckpt.get("value_gn_groups", 8))
        kw["value_pool"] = str(ckpt.get("value_value_pool", kw.get("value_pool", "gap")))
        m = create_model(**kw)
        m.load_state_dict(ckpt["value_state"])
    else:
        m = create_model(**model_kwargs_from_ckpt(ckpt))
        m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


def main():
    files = sorted((REPO / "training_history/az_fixed/warm_polonly").glob("*.npz"))
    B, S, HP = [], [], []
    for f in files[:60]:                      # ~50k states (float32 boards ~2GB)
        d = np.load(f)
        B.append(d["board"]); S.append(d["stats"]); HP.append(d["stats"][:, 1])
        d.close()
    B = torch.from_numpy(np.concatenate(B)).float()
    S = torch.from_numpy(np.concatenate(S)).float()
    HP = np.concatenate(HP).astype(np.float64)          # raw hp_delta (player view)
    N = B.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    tr, te = perm[: int(N * 0.8)], perm[int(N * 0.8):]
    print(f"states={N}  train={len(tr)} test={len(te)}")

    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = build_value_net(ck)
        with torch.no_grad():
            V = torch.cat([m(B[i:i + 512], S[i:i + 512])["value"].squeeze(-1)
                           for i in range(0, N, 512)]).numpy().astype(np.float64)
        # stats -> V probe
        probe = nn.Sequential(nn.Linear(42, 128), nn.ReLU(),
                              nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(probe.parameters(), lr=3e-3)
        Vt = torch.from_numpy(V).float()
        St = S
        for _ in range(300):
            idx = tr[torch.randint(len(tr), (1024,), generator=g)]
            loss = nn.functional.mse_loss(probe(St[idx]).squeeze(-1), Vt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            pred = probe(St[te]).squeeze(-1).numpy().astype(np.float64)
        r2 = 1 - np.var(V[te.numpy()] - pred) / np.var(V[te.numpy()])
        c = float(np.corrcoef(np.clip(HP / HP_SCALE, -1, 1), V)[0, 1])
        print(f"  {Path(path).name:34s} V std={V.std():.4f}  "
              f"stats-probe R2={r2:+.3f}  corr(V,hp_delta)={c:+.3f}")


if __name__ == "__main__":
    main()
