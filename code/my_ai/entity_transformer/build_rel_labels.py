#!/usr/bin/env python3
"""在**全量逐帧**序列上算「指数加权未来血量差」类标签，再按与 `subsample_frames.py`
**完全相同**的抽样（同一 seed / 同一逻辑）取那 64 帧，输出标签文件。

复用 `value_warmup._exp_weighted_labels`（`geo`/`kgeo`，`abs`/`rel`/`mix`），
它读 `stats[:,1]` 作为逐帧血量差，并按 `player` 统一到 P0 视角再回签。

标签语义（`rel`，推荐）：
    label_t = (指数加权未来血量差均值) − 当前帧血量差   ⇒ "从这位置出发未来优势会变多少"
部署时必须 `AZAI_VALUE_TANH=0 + AZAI_REL2ABS=1.0` 把绝对价值还原回来。

用法：
  python code/my_ai/entity_transformer/build_rel_labels.py \
      --src F:/.../lowq_e2_merged --out D:/.../lowq_f64 --tau 5 10 50 --weight geo kgeo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "code"))

from my_ai.az_intent.value_warmup import _exp_weighted_labels  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, nargs="+", default=[5.0, 10.0, 50.0])
    ap.add_argument("--weight", nargs="+", default=["geo", "kgeo"])
    ap.add_argument("--mode", default="rel", choices=["abs", "rel", "mix"])
    ap.add_argument("--frames-per-game", type=int, default=64)
    ap.add_argument("--save-full", action="store_true",
                    help="把**全量逐帧**标签也存下来（全量帧训练用）")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    src = Path(a.src)
    out = Path(a.out)
    offs = np.load(src / "game_offsets.npy")
    stats = np.load(src / "stats.npy", mmap_mode="r")
    player = np.load(src / "player.npy", mmap_mode="r")
    lab_term = np.load(src / "value_win.npy")
    n_games = len(offs) - 1
    K = a.frames_per_game
    print(f"[rel] {n_games} games, K={K}, mode={a.mode}", flush=True)

    variants = [(w, t) for w in a.weight for t in a.tau]
    # 全量逐帧标签缓存
    full = {}
    for (w, t) in variants:
        full[(w, t)] = np.zeros(len(stats), dtype=np.float32)

    # 逐局算全量标签
    for g in range(n_games):
        s0, e0 = int(offs[g]), int(offs[g + 1])
        st = np.asarray(stats[s0:e0])
        pl = np.asarray(player[s0:e0])
        for key in variants:
            w, t = key
            full[key][s0:e0] = _exp_weighted_labels(
                st, pl, tau=float(t), label_scale=1.0,
                label_mode=a.mode, label_weight=w)
        if (g + 1) % 1000 == 0:
            print(f"  [{g+1}/{n_games}]", flush=True)

    # 复刻 subsample_frames.py 的抽样（同 seed / 同逻辑）
    rng = np.random.default_rng(a.seed)
    fidx = np.empty(n_games * K, dtype=np.int64)
    wptr = 0
    for g in range(n_games):
        s0, e0 = int(offs[g]), int(offs[g + 1])
        n = e0 - s0
        if n <= K:
            idx = np.arange(s0, e0)
        else:
            edges = np.linspace(0, n, K + 1).astype(np.int64)
            idx = np.array([s0 + int(rng.integers(edges[k], edges[k + 1])) for k in range(K)])
        fidx[wptr:wptr + len(idx)] = idx
        wptr += len(idx)
    fidx = fidx[:wptr]
    np.save(out / "frame_idx.npy", fidx)

    term = lab_term[fidx]
    print(f"\n{'variant':>16} | {'mean':>7} {'std':>7} {'min':>6} {'max':>6} | corr(term)")
    for key in variants:
        w, t = key
        y = full[key][fidx]
        np.save(out / f"label_{w}_tau{int(t)}.npy", y)
        if a.save_full:
            np.save(out.parent / "lowq_full" / f"label_{w}_tau{int(t)}.npy", full[key])
        c = float(np.corrcoef(y, term)[0, 1])
        print(f"{w + '_tau' + str(int(t)):>16} | {y.mean():7.3f} {y.std():7.3f} "
              f"{y.min():6.2f} {y.max():6.2f} | {c:.3f}")
    print("\n[rel] 参考: terminal 标签 std=%.3f" % term.std(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
