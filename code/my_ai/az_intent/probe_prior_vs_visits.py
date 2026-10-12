"""探针：**候选的先验（PUCT 的 P）到底有没有进访问分配？**

背景（2026-10-12）：网格显示 `cands_prior=map`（图上概率 Π_头 p(格)）几乎不改变 topk 臂的 visits（0.113 vs 0.109）✗
⇒ 要么先验实际很平，要么它没主导分配。本探针直接读 `root.children[i].prior` 与 `children[i].visits` 并排打印。

用法：python -u code/my_ai/az_intent/probe_prior_vs_visits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "Ant-Game", _REPO / "code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    # ⚠️ 2026-10-12（踩坑修正）：`decoder.make_class_mask` 在**调用时**读 `AZAI_BAN_CLASSES`（decoder.py:167）
    #   ⇒ 探针/网格必须设成与冻结配置一致（禁闪电类 17 + op 21），否则会跑在"允许闪电"的制度里 ✗
    #   （我第一版就没设 ⇒ 所有候选都是 op 21 的闪电落点 ⇒ 与本子博弈无关 ✗）
    import os
    os.environ.setdefault("AZAI_BAN_CLASSES", "17")
    os.environ.setdefault("AZAI_BAN_OPS", "21")

    from SDK.utils.features import FeatureExtractor

    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    _model, net_fn = make_net_fn_from_ckpt("checkpoints/grafted_sdk_A.pt", feat)
    st = make_initial_state(90000, True)
    for _ in range(80):
        st.advance_round()

    def mks(seed, cands, prior, iters=256):
        return BundleMCTS(net_fn, iterations=iters, max_depth_rounds=4, k=24, sample_mult=15,
                          t_class=0.5, t_pos=1.0, c_puct=1.25, seed=seed, search_mode="pos-only",
                          pos_pin="mc_quiet", skip_single_candidate=True, mc_horizon=256, mc_every=1,
                          mc_root_only=1, mc_topk=3, mc_upg=1,
                          cands_mode=cands, cands_k=0, cands_combo=1, cands_prior=prior)

    for cands, prior in (("sample", "uniform"), ("topk", "uniform"), ("topk", "map")):
        m = mks(7, cands, prior)
        r = m.search(st, 0, temperature=1.0)
        root = getattr(m, "last_root", None)
        if root is None or not root.children:
            print(f"  {cands}/{prior}: 无 last_root.children（可能被 skip_single_candidate 短路）")
            continue
        rows = sorted(((c.visits, float(c.prior), i) for i, c in enumerate(root.children)),
                      key=lambda t: -t[0])
        vr = np.asarray([c.visits for c in root.children], dtype=np.float64)
        pr = np.asarray([float(c.prior) for c in root.children], dtype=np.float64)
        pv = vr / max(vr.sum(), 1e-9)
        pp = pr / max(pr.sum(), 1e-9)
        print(f"\n  === {cands} / prior={prior}：子 {len(root.children)} 个；"
              f"**先验 top1={pp.max():.3f} 熵={-(pp[pp>0]*np.log2(pp[pp>0])).sum():.2f}bit**；"
              f"**visits top1={pv.max():.3f} 熵={-(pv[pv>0]*np.log2(pv[pv>0])).sum():.2f}bit**")
        print("      (visits, prior, 子下标, ops) 前 8：")
        for v, p, i in rows[:8]:
            _ops = root.bundles[i] if i < len(root.bundles) else None
            print(f"        visits={v:5.0f}  prior={p:.6f}  child#{i}  ops={_ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
