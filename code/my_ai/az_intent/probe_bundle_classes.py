"""任务 1（零对局）：**拓宽菜单是否改变了"选中 bundle 的类组成"**？⇒ 检验"伤害经由类仲裁"这条机制 ✓。

对同一批状态跑两/三种配置，逐条打印 `argmax(visits)` 那个 bundle 的**完整 op 列表**（类、格、op 数 ✓），
并汇总：类直方图、平均 op 数、含类 16（降级）的比例、≥2 op 的比例 ✓；最后给两配置**选中类的一致率** ✓。
（逐条打印 ⇒ 即使跑一半也能读 ✓；K=100 较慢，用 `--states` 控制规模 ✓。）

用法：python -u code/my_ai/az_intent/probe_bundle_classes.py --states 8 --configs 24:256,100:2000
"""
from __future__ import annotations
import argparse, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--configs", default="24:256,100:2000")
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--rounds", default="80,160,240")
    ap.add_argument("--sample-mult", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17"); os.environ.setdefault("AZAI_BAN_OPS", "21")
    import torch
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    _m, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    cfgs = []
    for tok in args.configs.split(","):
        k, it = tok.split(":"); cfgs.append((int(k), int(it)))
    rounds = [int(x) for x in args.rounds.split(",")]
    per = max(1, args.states // max(len(rounds), 1))
    states = []
    for r in rounds:
        for i in range(per):
            st = make_initial_state(91000 + i * 7 + r, True)
            for _ in range(r):
                if st.terminal:
                    break
                st.advance_round()
            if not st.terminal:
                states.append(st)
    print(f"[cls] ckpt={Path(args.ckpt).name} 状态 {len(states)} 个；配置 {cfgs}", flush=True)

    chosen = {}
    for (k, it) in cfgs:
        hist = Counter(); sizes = []; has16 = 0; multi = 0; n = 0
        t0 = time.time()
        for i, st in enumerate(states):
            m = BundleMCTS(net_fn, iterations=it, max_depth_rounds=4, k=k, sample_mult=args.sample_mult,
                           t_class=0.5, t_pos=1.0, c_puct=1.25, seed=args.seed, search_mode="pos-only",
                           pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_every=1,
                           mc_root_only=1, mc_topk=3, mc_upg=1)
            m.search(st, 0, temperature=1e-6)
            root = getattr(m, "last_root", None)
            if root is None or len(root.children) < 2:
                continue
            v = np.asarray([float(c.visits) for c in root.children], dtype=np.float64)
            pick = int(np.argmax(v))
            ops = [tuple(int(x) for x in op) for op in root.bundles[pick]]
            classes = [op[0] for op in ops if len(op) >= 3]
            cells = [(op[1], op[2]) for op in ops if len(op) >= 3]
            n += 1
            sizes.append(len(ops))
            for c in classes:
                hist[c] += 1
            if 16 in classes:
                has16 += 1
            if len(ops) >= 2:
                multi += 1
            chosen.setdefault((k, it), {})[i] = (tuple(classes), tuple(cells))
            print(f"    [{k:>3d}/{it:>5d}] 状态 {i:>2d}: op={ops} ⇒ 类 {classes}（{len(ops)} 个 op）", flush=True)
        dt = time.time() - t0
        if n:
            print(f"  === K={k} ITERS={it}（有效 {n}；用时 {dt:.0f}s）类直方图 {dict(sorted(hist.items()))}；"
                  f"平均 op 数 {np.mean(sizes):.2f}；**含降级(类16) {has16/n*100:.1f}%**；≥2 op {multi/n*100:.1f}%", flush=True)
    if len(cfgs) >= 2:
        a, b = chosen[cfgs[0]], chosen[cfgs[-1]]
        common = sorted(set(a) & set(b))
        if common:
            agree = sum(1 for i in common if set(a[i][0]) == set(b[i][0])) / len(common) * 100
            same_cells = 0
            ious = []
            for i in common:
                sa, sb = set(a[i][1]), set(b[i][1])
                if sa == sb:
                    same_cells += 1
                if sa or sb:
                    ious.append(len(sa & sb) / max(len(sa | sb), 1))
            print(f"\n[cls] 基线与最宽配置：**选中类集合一致率 = {agree:.1f}%**；"
                  f"**格集合完全相同 {same_cells / len(common) * 100:.1f}%**；格集合 IoU 均值 {np.mean(ious) * 100:.1f}%"
                  f"（{len(common)} 个可比状态）", flush=True)
            print("       判读：类一致率高 + 格 IoU 低 ⇒ 伤害在『同类但不同格/不同小计划』（机制② ✓）；"
                  "类一致率低 ⇒ MC 类仲裁被扰动（机制③ ✓）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
