"""诊断：**搜索的"不可复现"主要来自"候选生成随机" 还是"搜索内部随机"？**（零对局）

做法：用真实中局状态（短自对弈中途快照），对**同一个局面同一个玩家**跑两次搜索（只换 `seed`），比较：
  * **候选束集合**是否完全相同（集合 = 候选束的集合，束 = op 三元组元组）；
  * **选中束**是否相同（用 `visit_policy` 的 argmax；`temperature=1.0` ⇒ policy ∝ 原始访问计数，见 `_policy_from_visits`）；
  * **访问分布余弦**（在两遍候选的并集上对齐）。

三档"候选生成随机性"（`t_class`/`t_pos` 就是 `decode_head` 的类/位置采样温度，**部署配置是 0.5/1.0**）：
  A 部署温度 0.5 / 1.0      ⇒ 候选束每次都重采（同部署）
  B 类确定 1e-6 / 1.0       ⇒ 类不再随机，位置仍采样
  C 全确定 1e-6 / 1e-6      ⇒ 候选生成几乎确定性

判读：
  * 若 A 的"候选集合相同率/选中一致率"很低，而 C 明显变高 ⇒ **主因是候选生成（采样）的随机性**；
  * 若三档都低 ⇒ 主因在**搜索内部**（值估计噪声 / PUCT 并列 / 展开采样）。

用法：python -u code/my_ai/az_intent/diag_search_repro.py --games 3 --snapshots 40,80,120,160
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
    ap.add_argument("--out", default="training_history/posroll2/diag_search_repro.json")
    args = ap.parse_args()

    import torch
    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.decoder import decode_network_output

    feat = FeatureExtractor(max_actions=96)
    _model, net_fn = make_net_fn_from_ckpt(args.ckpt, feat)
    snaps = [int(x) for x in args.snapshots.split(",")]

    def make_mcts(seed: int, tc: float, tp: float) -> BundleMCTS:
        return BundleMCTS(
            net_fn, iterations=256, max_depth_rounds=4, k=24, sample_mult=15,
            t_class=tc, t_pos=tp, c_puct=1.25, seed=seed, search_mode="pos-only",
            pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_bidir=0,
            mc_every=1, mc_no_downgrade=0, mc_m4=0, mc_root_only=1, mc_top2=0,
            mc_topk=3, mc_cells=0, mc_upg=1, mc_margin=0.0, mc_score="",
            mc_selfspend=0, mc_selfspend_mix=1.0,
        )

    # 1) 造中局状态：**只推进回合**（双方都不落子 ⇒ 无人建塔，棋盘上有蚂蚁/金币/血量变化）
    #    ⚠️ 口径：这些状态里没有塔 ⇒ 候选以"建造"为主；这是**相对分解**测量，不是"真实中局分布"。
    states = []
    for g in range(args.games):
        st = make_initial_state(args.seed0 + g, True)
        for rnd in range(1, max(snaps) + 1):
            if st.terminal:
                break
            if rnd in snaps:
                states.append(st.clone())
            st.advance_round()
    print(f"[repro] 造出 {len(states)} 个中局状态（{args.games} 局 × 快照 {snaps}）", flush=True)

    def run(state, player, seed, tc, tp):
        m = make_mcts(seed, tc, tp)
        r = m.search(state, player, temperature=1.0)     # >1e-6 ⇒ policy ∝ 原始访问计数
        bundles = [tuple(tuple(int(v) for v in op) for op in b) for b in r.bundles]
        w = np.asarray(r.visit_policy, dtype=np.float64)
        return bundles, w

    arms = [("A 部署温度 0.5/1.0", (0.5, 1.0)),
            ("B 类确定 1e-6/1.0", (1e-6, 1.0)),
            ("C 全确定 1e-6/1e-6", (1e-6, 1e-6))]
    out = {}
    for name, (tc, tp) in arms:
        n = eq = agr = agr_eq = n_eq = agr_neq = n_neq = 0
        jac, cos, sizes = [], [], []
        for st in states:
            b1, w1 = run(st, 0, 101, tc, tp)
            b2, w2 = run(st, 0, 202, tc, tp)
            n += 1
            s1, s2 = set(b1), set(b2)
            same = (s1 == s2)
            eq += int(same); n_eq += int(same); n_neq += int(not same)
            a1 = b1[int(np.argmax(w1))] if len(w1) else None
            a2 = b2[int(np.argmax(w2))] if len(w2) else None
            ag = int(a1 is not None and a1 == a2)
            agr += ag
            if same:
                agr_eq += ag
            else:
                agr_neq += ag
            union = s1 | s2
            jac.append(len(s1 & s2) / max(len(s1 | s2), 1))
            sizes.append((len(b1), len(b2)))
            if len(union) > 1:
                d1, d2 = dict(zip(b1, w1)), dict(zip(b2, w2))
                v1 = np.array([d1.get(k, 0.0) for k in union]); v2 = np.array([d2.get(k, 0.0) for k in union])
                if v1.sum() > 0 and v2.sum() > 0:
                    cos.append(float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)))
        out[name] = dict(n=n, sets_identical=eq, chosen_agree=agr,
                         chosen_agree_given_same=agr_eq, n_same=n_eq,
                         chosen_agree_given_diff=agr_neq, n_diff=n_neq,
                         jaccard_median=float(np.median(jac)) if jac else None,
                         cosine_median=float(np.median(cos)) if cos else None,
                         sizes_mean=[float(np.mean([s[0] for s in sizes])), float(np.mean([s[1] for s in sizes]))])
        d = out[name]
        print(f"  {name}: 候选集合完全相同 {d['sets_identical']}/{n}；"
              f"**选中一致 {d['chosen_agree']}/{n}**"
              f"（集合相同时 {d['chosen_agree_given_same']}/{max(d['n_same'],1)}，"
              f"集合不同时 {d['chosen_agree_given_diff']}/{max(d['n_diff'],1)}）；"
              f"Jaccard 中位 {d['jaccard_median']}；余弦中位 {d['cosine_median']}；"
              f"候选数均值 {d['sizes_mean'][0]:.1f} vs {d['sizes_mean'][1]:.1f}", flush=True)

    import json
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[repro] 摘要：{args.out}")
    print("[repro] 判读：A 低而 C 明显变高 ⇒ 主因是**候选生成采样**；三档都低 ⇒ 主因在**搜索内部**。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
