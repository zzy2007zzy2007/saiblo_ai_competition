"""先行检验：把位置目标的"量"从 **N（访问计数）** 换成 **Q = mean_value（回传均值）** 值不值得？

依据 `docs/az_q_target_plan.md` §2/§3：
  * 指标（该计划）：两遍搜索的 **Q 向量余弦**、top-5 重叠、Q 的 argmax 与 N 的 argmax 是否一致；
    参照 = **同一设定下 N 的余弦**（旧记录 0.35，但**条件不明** ✗ ⇒ 本脚本**在同一次运行里重测 N 基线** ✓）；
    绿灯 Q 余弦 ≥0.7 ✓ / 红灯 Q 和 N 一样烂 ⇒ 病根在"价值并列"，加深目标救不了 ✗。
  * **本脚本额外的第一条诊断（更重要）**：**同一遍搜索内** Q 与 N 的余弦 —— 若二者近乎共线，
    则"换 Q"**不带来新信息** ✗（可省掉整条采集路线 ✓）。

做法：在禁闪电制度下取若干中局状态（默认 `--games 3 --snapshots 80,160` ⇒ 6 个），对每个状态跑两遍搜索（不同 seed），
取 `m.last_root.children` 的 `(visits, mean_value)`；只保留**根节点有 ≥2 个子节点**的状态（否则无偏好可言 ✓）。
候选集合不同的情况按**并集**对齐（缺失置 0），与 `diag_search_repro.py` 同构 ✓。

用法：python -u code/my_ai/az_intent/diag_q_vs_n_repro.py --games 3 --snapshots 80,160
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


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float(a @ b / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/grafted_sdk_A.pt")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--snapshots", default="80,160")
    ap.add_argument("--seed0", type=int, default=90000)
    ap.add_argument("--iters", type=int, default=256)
    args = ap.parse_args()

    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")   # 制度卫生（见日志 86r/86y）
    os.environ.setdefault("AZAI_BAN_OPS", "21")

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

    def mks(seed: int) -> BundleMCTS:
        return BundleMCTS(net_fn, iterations=args.iters, max_depth_rounds=4, k=24, sample_mult=15,
                          t_class=0.5, t_pos=1.0, c_puct=1.25, seed=seed, search_mode="pos-only",
                          pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_every=1,
                          mc_root_only=1, mc_topk=3, mc_upg=1)

    def run(st, seed):
        m = mks(seed)
        m.search(st, 0, temperature=1.0)
        root = getattr(m, "last_root", None)
        if root is None or len(root.children) < 2:
            return None
        n = np.asarray([float(c.visits) for c in root.children], dtype=np.float64)
        q = np.asarray([float(c.mean_value) for c in root.children], dtype=np.float64)
        keys = [tuple(tuple(int(v) for v in op) for op in b) for b in root.bundles]
        return keys, n, q

    within_cos, within_argmax = [], []
    cos_n, cos_q, top5_n, top5_q, n_used = [], [], [], [], 0
    for st in states:
        r1 = run(st, 7)
        r2 = run(st, 99)
        if r1 is None or r2 is None:
            continue
        n_used += 1
        k1, n1, q1 = r1
        k2, n2, q2 = r2
        # ① 同一遍搜索内 Q vs N（关键诊断：Q 是否只是 N 的共线复制？）
        within_cos.append(_cos(n1, q1))
        within_argmax.append(int(np.argmax(n1) == np.argmax(q1)))
        # ② 跨两遍（并集对齐）—— N 与 Q 各自的可复现性，同一次运行里互为对照
        union = list(dict.fromkeys(k1 + k2))
        d1n, d1q = dict(zip(k1, n1)), dict(zip(k1, q1))
        d2n, d2q = dict(zip(k2, n2)), dict(zip(k2, q2))
        v1n = np.array([d1n.get(k, 0.0) for k in union]); v2n = np.array([d2n.get(k, 0.0) for k in union])
        v1q = np.array([d1q.get(k, 0.0) for k in union]); v2q = np.array([d2q.get(k, 0.0) for k in union])
        cos_n.append(_cos(v1n, v2n))
        cos_q.append(_cos(v1q, v2q))
        s1n = {k for k, _ in sorted(d1n.items(), key=lambda kv: -kv[1])[:5]}
        s2n = {k for k, _ in sorted(d2n.items(), key=lambda kv: -kv[1])[:5]}
        s1q = {k for k, _ in sorted(d1q.items(), key=lambda kv: -kv[1])[:5]}
        s2q = {k for k, _ in sorted(d2q.items(), key=lambda kv: -kv[1])[:5]}
        top5_n.append(len(s1n & s2n) / max(len(s1n | s2n), 1))
        top5_q.append(len(s1q & s2q) / max(len(s1q | s2q), 1))

    if not n_used:
        print("[qvn] 没有可用状态（根节点都只有 1 个子节点？）")
        return 0
    print(f"[qvn] 状态 {len(states)} 个，可用（根节点 ≥2 子且两遍都成功）{n_used} 个；iters={args.iters}")
    print(f"  ① **同一遍内 Q vs N**：余弦中位 {np.median(within_cos):.3f}；"
          f"两者的 argmax 一致率 {np.mean(within_argmax) * 100:.1f}%"
          f"   ← 若接近 1.0 ⇒ **Q ≈ N 的共线复制，换 Q 无新信息** ✗")
    print(f"  ② **跨两遍（不同 seed，并集对齐）**：")
    print(f"     N（visit）：余弦中位 {np.nanmedian(cos_n):.3f}；top-5 Jaccard 中位 {np.median(top5_n):.3f}")
    print(f"     **Q（mean_value）：余弦中位 {np.nanmedian(cos_q):.3f}；top-5 Jaccard 中位 {np.median(top5_q):.3f}**")
    print(f"     判读（计划 §3）：Q 余弦 ≥0.7 且明显高于 N ⇒ 绿灯，值得做采集 ✓；"
          f"否则红灯：病根在**价值并列**，加深目标救不了 ✗")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
