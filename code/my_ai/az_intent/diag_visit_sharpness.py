"""诊断：**真实访问分布有多"清晰"**（`visits_raw`，不是被 argmax 的 `visit_policy`）。

动机（2026-10-11）：`visit_policy` 在部署温度 `TEMP=1e-6` 下被 `_policy_from_visits` argmax 成 one-hot ✗，
我此前据此说"教师全平/单点退化"都是错的；单状态抽查显示真分布 **top1 份额 0.512、熵 2.996 bit（24 候选）** ✓。
本脚本把它量成统计量：对若干中局状态，分别用 **现状采样** 与 **topk 确定化** 生成候选，报告：
  * `visits_raw` 的 **top1 份额** 与 **归一化熵**（bit；并列 log2(候选数) 作参考）；
  * 两次搜索（换 seed）在 **相同候选集合** 下的余弦（topk 期望 ≈1）；
  * 顺带报告候选数与 `visits_raw` 的和（应 ≈ iterations=256）。

用法：python -u code/my_ai/az_intent/diag_visit_sharpness.py --games 3 --snapshots 40,80,120,160
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--snapshots", default="40,80,120,160")
    ap.add_argument("--seed0", type=int, default=90000)
    ap.add_argument("--iters", type=int, default=256)
    ap.add_argument("--combo", type=int, default=1, help="topk 模式：1=各头拼束；0=只用 head0（单 op）")
    ap.add_argument("--cands-prior", default="uniform", choices=["uniform", "map"],
                    help="topk 候选的先验：uniform=每个候选 count=1（PUCT 先验均匀 ✗）；map=图上概率 Π_头 p(格)（公平对照 ✓）")
    args = ap.parse_args()

    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    _model, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    snaps = [int(x) for x in args.snapshots.split(",")]

    states = []
    for g in range(args.games):
        st = make_initial_state(args.seed0 + g, True)
        for rnd in range(1, max(snaps) + 1):
            if st.terminal:
                break
            if rnd in snaps:
                states.append(st.clone())
            st.advance_round()
    print(f"[sharp] 状态 {len(states)} 个（只推进回合、无塔 ⇒ 候选以建造为主，相对测量）", flush=True)

    def make_mcts(seed: int, cands: str) -> BundleMCTS:
        return BundleMCTS(net_fn, iterations=args.iters, max_depth_rounds=4, k=24, sample_mult=15,
                          t_class=0.5, t_pos=1.0, c_puct=1.25, seed=seed, search_mode="pos-only",
                          pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_every=1,
                          mc_root_only=1, mc_topk=3, mc_upg=1,
                          cands_mode=cands, cands_k=0, cands_combo=args.combo,
                          cands_prior=args.cands_prior)

    import time as _time

    for cands in (("topk",) if args.combo == 0 else ("sample", "topk")):
        top1, ent, ncand, sums, cos, mrank, secs = [], [], [], [], [], [], []
        for st in states:
            _t0 = _time.time()
            r = make_mcts(7, cands).search(st, 0, temperature=1e-6)   # 部署口径
            secs.append(_time.time() - _t0)
            vr = np.asarray(r.visits_raw, dtype=np.float64)
            if vr.sum() <= 0:
                continue
            p = vr / vr.sum()
            top1.append(float(p.max()))
            ent.append(float(-(p[p > 0] * np.log2(p[p > 0])).sum()))
            ncand.append(len(vr))
            sums.append(float(vr.sum()))
            # 新指标（2026-10-12，用户追问逼出）：**搜索的 pick（visits 的 argmax）在"按图分排序"里的名次**
            #   rank=1 ⇒ 搜索选的就是图上最好的那一格；名次越大越"不贴图" ⇒ 直接预测部署强度 ✓
            try:
                am = np.asarray(net_fn(st, 0)["action_map"], dtype=np.float64)
                keys = [tuple(tuple(int(v) for v in op) for op in b) for b in r.bundles]
                vals = np.array([am[int(k[0][0]), int(k[0][1]), int(k[0][2])] for k in keys])
                order = np.argsort(-vals, kind="stable")
                pick = int(np.argmax(vr))
                mrank.append(int(np.where(order == pick)[0][0]) + 1)
            except Exception:  # noqa: BLE001
                pass
            r2 = make_mcts(99, cands).search(st, 0, temperature=1e-6)
            b1 = {tuple(tuple(int(v) for v in op) for op in b) for b in r.bundles}
            b2 = {tuple(tuple(int(v) for v in op) for op in b) for b in r2.bundles}
            m1 = {k: v for k, v in zip([tuple(tuple(int(v) for v in op) for op in b) for b in r.bundles], vr)}
            v2 = np.asarray(r2.visits_raw, dtype=np.float64)
            m2 = {k: v for k, v in zip([tuple(tuple(int(v) for v in op) for op in b) for b in r2.bundles], v2)}
            if b1 == b2 and len(b1) > 1:
                u = list(b1)
                a = np.array([m1[k] for k in u]); b = np.array([m2[k] for k in u])
                if a.sum() > 0 and b.sum() > 0:
                    cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
        if not top1:
            print(f"  {cands}: 无有效样本")
            continue
        _r1 = np.mean([x == 1 for x in mrank]) * 100 if mrank else float("nan")
        _rm = np.median(mrank) if mrank else float("nan")
        print(f"  **{cands}**（iters={args.iters}，prior={args.cands_prior}）: 候选数 {np.mean(ncand):.1f}；"
              f"visits_raw 和 {np.mean(sums):.1f}；"
              f"**top1 份额 {np.median(top1):.3f}**（均值 {np.mean(top1):.3f}）；"
              f"**归一化熵 {np.median(ent):.3f} bit**（并列 log2={np.log2(np.mean(ncand)):.2f}）；"
              f"**搜索 pick 的图排名 rank1 {_r1:.1f}% / 中位 {_rm:.0f}**；"
              f"同候选集余弦 {'—' if not cos else f'{np.median(cos):.6f}'}（N={len(cos)}）；"
              f"单次 {np.mean(secs):.2f}s", flush=True)
    print("[sharp] 判读：top1 份额明显 >1/n 且熵明显 <log2(n) ⇒ **访问分布是清晰的偏好**（不是退化/全平）✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
