"""实测：把候选菜单拓宽（`K`）与提高迭代数（`ITERS`）的**代价与形态**。

背景（用户 2026-10-12 提议）：保留原动作网，把候选范围扩到 ~100，并同步提高迭代数，
使 **ITERS / K ≈ 10–20**（当前 `256/24 ≈ 10.7` ✓ 已在区间内 ⇒ 实质改变是"**菜单更宽、每候选同样访问数**" ✓）。

本脚本只回答三件事（零对局 ✓，禁闪电制度 ✓）：
  1. **实际保留的候选数**（`len(root.children)` ✓ —— 注意它是"按采样次数取 top-k" ✓，K 只是上限 ✓）；
  2. **访问数分布**：`sum(visits)`、每候选平均访问、top-1 占比 ✓；
  3. **单次决策耗时** ⇒ 据此推算 128 局判据臂的墙钟代价 ✓（决定要不要投 ✓）。

用法：
  python -u code/my_ai/az_intent/probe_k_iters.py --ks 24,100 --iters 256,1000,1500,2000 --states 2
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--ks", default="24,100")
    ap.add_argument("--iters", default="256,1000,1500,2000")
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--round", type=int, default=120)
    ap.add_argument("--sample-mult", type=int, default=15)
    args = ap.parse_args()

    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")
    os.environ.setdefault("AZAI_BAN_OPS", "21")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    _m, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)

    states = []
    for i in range(args.states):
        st = make_initial_state(91000 + i, True)
        for _ in range(args.round):
            if st.terminal:
                break
            st.advance_round()
        states.append(st)

    print(f"[probe] ckpt={Path(args.ckpt).name} 状态 {len(states)} 个（第 {args.round} 回合）"
          f"，sample_mult={args.sample_mult}", flush=True)
    print(f"  {'K':>4s} {'ITERS':>6s} {'候选数':>6s} {'sum(v)':>7s} {'每候选':>7s} {'top1占比':>8s} "
          f"{'不同格':>7s} {'单次耗时':>9s} {'推算 128 局':>12s}", flush=True)
    for K in [int(x) for x in args.ks.split(",")]:
        for IT in [int(x) for x in args.iters.split(",")]:
            rows = []
            for st in states:
                t0 = time.time()
                m = BundleMCTS(net_fn, iterations=IT, max_depth_rounds=4, k=K,
                               sample_mult=args.sample_mult, t_class=0.5, t_pos=1.0, c_puct=1.25,
                               seed=7, search_mode="pos-only", pos_pin="mc_quiet",
                               skip_single_candidate=True, mc_horizon=256, mc_every=1,
                               mc_root_only=1, mc_topk=3, mc_upg=1)
                m.search(st, 0, temperature=1.0)
                dt = time.time() - t0
                root = getattr(m, "last_root", None)
                if root is None or not len(root.children):
                    rows.append((dt, 0, 0.0, 0.0, 0.0, 0))
                    continue
                v = np.asarray([float(c.visits) for c in root.children], dtype=np.float64)
                labels = root.bundles
                cells = {(int(op[1]), int(op[2])) for b in labels for op in b if len(op) >= 3}
                rows.append((dt, len(root.children), float(v.sum()),
                             float(v.mean()), float(v.max() / max(v.sum(), 1e-9)), len(cells)))
            nc = np.mean([r[1] for r in rows])
            sv = np.mean([r[2] for r in rows])
            pc = np.mean([r[3] for r in rows])
            t1 = np.mean([r[4] for r in rows]) * 100
            cd = np.mean([r[5] for r in rows])
            dt = np.mean([r[0] for r in rows])
            # 128 局 ≈ 每局 ~75 次"搜索真正起作用"的决策（15% × ~490）+ 其余决策的 MC 成本已含在现基线里；
            # 这里只按"决策次数 × 单次耗时"粗估，用于筛掉不可行的档位 ✓
            est_h = 128 * 75 * dt / 8 / 3600
            print(f"  {K:>4d} {IT:>6d} {nc:>6.1f} {sv:>7.1f} {pc:>7.2f} {t1:>7.1f}% "
                  f"{cd:>7.1f} {dt:>8.2f}s {est_h:>10.1f}h", flush=True)
    print("[probe] 判读：① 候选数是否真能接近 K（受『按采样次数取 top-k』限制 ✓）；"
          "② 每候选访问是否仍在 10–20 ✓；③ 推算耗时能否接受（>8h 要慎重 ✓）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
