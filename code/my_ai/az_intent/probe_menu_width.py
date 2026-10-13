"""零对局预检：**拓宽候选菜单（K）到底会不会改变搜索的选择？**（决定要不要投 128 局判据臂）

用户 2026-10-12 的假设：AZ 里策略网的职责是"**把好动作筛出来**" ⇒ 若给足算力 + 拓宽候选后强度不变，
则"筛"这一步没亏 ⇒ 动作网在**支撑集**这一侧没有多少优化空间 ✓。
本脚本把该假设中"需要打对局"的部分拆掉：先在**零对局**下看
"**宽菜单与窄菜单最终选的格是否不同**" ✓ —— 若几乎总相同，判据臂大概率也是平的 ⇒ 省下 6 h ✓。

对每个状态 × 每个 (K, ITERS)：
  * **单次决策耗时**（含 `n_samples = K × sample_mult` 的采样成本 ✓）；
  * 实际保留候选数 `len(root.children)`、`sum(visits)`、**每候选访问数**（用户要求 ≈10–20 ✓）、top-1 占比、不同格数；
  * **选中格**（`argmax(visits)` 那个 bundle 的第一个带位置 op 的 (x,y) ✓）；
  * 与基线（第一个配置，默认 K=24/ITERS=256）的**一致率** ✓，以及"基线选择在宽菜单里的名次" ✓。

用法：
  python -u code/my_ai/az_intent/probe_menu_width.py --states 6 --rounds 60,120,200 \
      --configs 24:256,50:550,100:1100,100:2000
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


def _cell_of(bundle):
    for op in bundle:
        if len(op) >= 3:
            return int(op[1]), int(op[2])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--configs", default="24:256,50:550,100:1100,100:2000")
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--rounds", default="60,120,200")
    ap.add_argument("--sample-mult", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")   # 制度卫生 ✓
    os.environ.setdefault("AZAI_BAN_OPS", "21")
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    _m, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)

    cfgs = []
    for tok in args.configs.split(","):
        k, it = tok.split(":")
        cfgs.append((int(k), int(it)))
    base = cfgs[0]
    rounds = [int(x) for x in args.rounds.split(",")]

    states = []
    per = max(1, args.states // max(len(rounds), 1))
    for r in rounds:
        for i in range(per):
            st = make_initial_state(91000 + i * 7 + r, True)
            for _ in range(r):
                if st.terminal:
                    break
                st.advance_round()
            if not st.terminal:
                states.append(st)
    print(f"[probe] ckpt={Path(args.ckpt).name}；状态 {len(states)} 个；配置 {cfgs}；sample_mult={args.sample_mult}",
          flush=True)

    def search(st, k, it):
        t0 = time.time()
        m = BundleMCTS(net_fn, iterations=it, max_depth_rounds=4, k=k, sample_mult=args.sample_mult,
                       t_class=0.5, t_pos=1.0, c_puct=1.25, seed=args.seed, search_mode="pos-only",
                       pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_every=1,
                       mc_root_only=1, mc_topk=3, mc_upg=1)
        m.search(st, 0, temperature=1e-6)
        dt = time.time() - t0
        root = getattr(m, "last_root", None)
        if root is None or not len(root.children):
            return dt, None, None
        v = np.asarray([float(c.visits) for c in root.children], dtype=np.float64)
        return dt, [_cell_of(b) for b in root.bundles], v

    base_choice = {}
    for i, st in enumerate(states):
        _dt, cells, v = search(st, *base)
        if cells:
            base_choice[i] = cells[int(np.argmax(v))]
    print(f"   基线 K={base[0]} ITERS={base[1]}：有选择的状态 {len(base_choice)}/{len(states)}"
          f"（其余是该回合只有单候选 ⇒ 搜索短路 ✓）", flush=True)

    print(f"\n  {'K':>4s} {'ITERS':>6s} {'候选数':>6s} {'sum(v)':>7s} {'每候选':>7s} {'top1':>6s} "
          f"{'不同格':>7s} {'耗时/次':>8s} {'与基线同格':>10s} {'基线选择名次':>13s} {'推算128局':>10s}", flush=True)
    for (k, it) in cfgs:
        rows, same, ranks = [], 0, []
        for i, st in enumerate(states):
            dt, cells, v = search(st, k, it)
            if not cells:
                continue
            pick = int(np.argmax(v))
            rows.append((dt, len(cells), float(v.sum()), float(v.mean()),
                         float(v.max() / max(v.sum(), 1e-9)), len({c for c in cells if c})))
            bc = base_choice.get(i)
            if bc is not None:
                if cells[pick] == bc:
                    same += 1
                for r, j in enumerate(np.argsort(-v, kind="stable"), 1):
                    if cells[j] == bc:
                        ranks.append(r)
                        break
        if not rows:
            print(f"  {k:>4d} {it:>6d}  （无有效状态）", flush=True)
            continue
        nc = np.mean([r[1] for r in rows]); sv = np.mean([r[2] for r in rows])
        pc = np.mean([r[3] for r in rows]); t1 = np.mean([r[4] for r in rows]) * 100
        cd = np.mean([r[5] for r in rows]); dt = np.mean([r[0] for r in rows])
        est = 128 * 75 * dt / 8 / 3600
        rk = f"{np.mean(ranks):.1f}（中位 {np.median(ranks):.0f}）" if ranks else "-"
        den = max(len(ranks), 1)
        print(f"  {k:>4d} {it:>6d} {nc:>6.1f} {sv:>7.1f} {pc:>7.2f} {t1:>5.1f}% {cd:>7.1f} "
              f"{dt:>7.2f}s {same / den * 100:>9.1f}% {rk:>13s} {est:>8.1f}h", flush=True)

    print("\n[probe] 判读：")
    print("  * 候选数是否真接近 K（受『按采样次数取 top-k』限制 ✓）；每候选访问是否仍在 10–20 ✓；")
    print("  * 若『与基线同格』很高（>90%）⇒ 拓宽菜单几乎不改变选择 ⇒ 判据臂大概率平 ⇒ 省下 6 h ✓；")
    print("  * 若明显更低（<80%）⇒ 菜单确实在改决策 ⇒ 值得投 128 局判据臂 ✓（并看基线的名次分布 ✓）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
