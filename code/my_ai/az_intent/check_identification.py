"""离线机制判据 P-M3a / P-M3b（预注册 `docs/prereg_20261005_exploration_identification.md` §3.1）。

**不需要跑对局**：
  * **P-M3a（识别性）**：把 `Δ = Q(花钱类) − Q(HOLD)` 按**金币**分组，看 Δ 与金币是否**正相关**、
    以及"穷"组里 Δ 是否 ≤0。旧数据（M2）的 Q 头在所有金币档里 Δ 都 > 0（= 花钱到处都显得好）。
  * **P-M3b（近局面多类覆盖）**：同一局内相邻决策视为"近局面"，统计"相邻决策里出现过 ≥2 个不同被执行类"的比例。

用法:
  python -u code/my_ai/az_intent/check_identification.py \
      --data training_history/vprior/data_sp_exp200_exec \
      --q training_history/vprior/qhead_M3.pt \
      --ckpt training_history/vprior/posnet_A_k5_m32.pt
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
for _p in (REPO / "Ant-Game", REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# stats 里金币的位置：az_selfplay 的 `g0/g1 = state.coins` 之外，特征工程里 coins 的索引未知，
# 这里改用**从 pkl 里能直接拿到的**代理：`stats` 的 [0..41] 中与金币最相关的分量由调用者指定。
# 为了稳妥，本脚本优先读 pkl 里的 `stats`，并用 `--coin-idx` 指定（默认 1 = HP，会被覆盖）。
DEFAULT_COIN_IDX = -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--q", default="")
    ap.add_argument("--ckpt", default="training_history/vprior/posnet_A_k5_m32.pt")
    ap.add_argument("--coin-idx", type=int, default=DEFAULT_COIN_IDX,
                    help="stats 向量里金币所在下标；未指定时自动探测")
    ap.add_argument("--limit-games", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.data)
    if not src.is_absolute():
        src = REPO / src
    files = sorted(src.glob("az_selfplay_seed*.pkl"))
    if a.limit_games:
        files = files[:a.limit_games]
    if not files:
        raise SystemExit(f"{src} 下没有 pkl")

    rows = []
    multi = 0
    games = 0
    for f in files:
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        s = d["samples"] if isinstance(d, dict) else d
        games += 1
        prev = None
        for r in s:
            cc = [int(x) for x in np.asarray(r["chosen_cls"])]
            rows.append((np.asarray(r["board"], dtype=np.float16),
                         np.asarray(r["stats"], dtype=np.float32), cc))
            if prev is not None and len(set(cc) | set(prev)) >= 2:
                multi += 1
            prev = cc
    print(f"局数={games} 决策={len(rows)}")
    print(f"[P-M3b] 相邻决策里出现过的**不同被执行类**≥2 的比例 = {multi/len(rows):.1%}"
          f"（旧数据 0.8% 的那种是按 head 计的，这里按决策计，量级可比）")

    # 金币索引：stats 里通常有 coins；先看看 stats 的分布，选方差最大且量级像金币的那一维
    st = np.stack([r[1] for r in rows])
    print("[stats] 各维均值/标准差（前 12 维）:")
    for i in range(min(12, st.shape[1])):
        print(f"   [{i:2d}] mean={st[:, i].mean():+8.2f} std={st[:, i].std():8.2f}")

    coin_idx = a.coin_idx
    if coin_idx < 0:
        # 自动：在所有维里找"取值大多在 0..600、均值几十~几百"的那一维（金币特征）
        cand = []
        for i in range(st.shape[1]):
            m, sd = float(st[:, i].mean()), float(st[:, i].std())
            if 5 <= m <= 800 and sd > 10:
                cand.append((i, m, sd))
        if cand:
            coin_idx = max(cand, key=lambda t: t[1])[0]
        else:
            coin_idx = 0
        print(f"[coin] 自动探测金币维 = {coin_idx}")

    if not a.q:
        print("（未给 --q，跳过 P-M3a）")
        return 0

    from my_ai.az_intent.az_selfplay import load_three_models
    from my_ai.az_intent.q_head import QHead

    q = QHead.load(str((REPO / a.q) if not Path(a.q).is_absolute() else a.q))
    ck = str((REPO / a.ckpt) if not Path(a.ckpt).is_absolute() else a.ckpt)
    class_model, _p, _v = load_three_models(ck)
    class_model.eval()

    idx = np.arange(0, len(rows), max(1, len(rows) // 4000))
    b = np.stack([rows[i][0] for i in idx]).astype(np.float32)
    s = np.stack([rows[i][1] for i in idx]).astype(np.float32)
    embs = np.zeros((len(idx), 128), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), 512):
            o = class_model(torch.from_numpy(b[i:i + 512]), torch.from_numpy(s[i:i + 512]))
            embs[i:i + 512] = o["state_emb"].cpu().numpy()
    with torch.no_grad():
        qq = q.score_all(torch.from_numpy(embs)).cpu().numpy()      # (M, 24)
    delta = qq[:, 0:17].max(axis=1) - qq[:, 23]                     # 最值钱的花钱类 vs HOLD
    coins = s[:, coin_idx]

    print(f"\n[P-M3a] n={len(idx)}；Δ=Q(最好花钱类)−Q(HOLD) 与金币({coin_idx}) 的相关 = "
          f"{float(np.corrcoef(delta, coins)[0,1]):+.3f}")
    qs = np.quantile(coins, [0, 0.25, 0.5, 0.75, 1.0])
    print("  按金币四分位:")
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (coins >= lo) & (coins <= hi)
        if m.any():
            print(f"    coins∈[{lo:7.1f},{hi:7.1f}] n={int(m.sum()):5d}  "
                  f"Δ均值={delta[m].mean():+.4f}  Δ>0 比例={float((delta[m] > 0).mean()):5.1%}")
    print("  判读：新数据若 Δ 随金币上升、且穷组 Δ<=0 ⇒ 识别性变好；旧数据（M2）是全档 Δ>0。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
