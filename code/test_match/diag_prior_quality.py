"""诊断先验质量：先验排序 vs rollout 估值排序（2026-10-23 用户提出的核心问题）。

问题：候选筛选（动作类头 + 位置网）基本是坏的（除闪电通道），
      我们一直在用"广度 k=160"硬补。这里量化这个补偿到底有多大代价。

做法（对同一局面，枚举全部候选，最多 ~266 个）：
  1. 取 _expand 生成的全候选，记下**先验**（bundle 的 prior / net 输出）；
  2. 对每个候选跑一次 h=16 rollout，得到**估值**；
  3. 计算：
     · 先验 vs 估值的 Spearman 相关（先验准不准）
     · 只用先验 top-k 时，"估值最高的那个动作"是否还在里面（recall@k）
     · 估值最高 vs 先验 top-1 的分数差（先验错选要付多少代价）
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from SDK.utils.features import FeatureExtractor
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else 0.0


def main() -> int:
    import sys as _sys
    ckpt = _sys.argv[1] if len(_sys.argv) > 1 else "checkpoints/posnet_A_k5_m32.pt"
    print(f"  ckpt = {ckpt}")
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(ckpt, feat, value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(ckpt)
    vm.eval()

    for seed, warm in ((1540, 60), (1540, 140), (1541, 200)):
        f = GameStateFacade.initial(seed=seed)
        for _ in range(warm):
            f.advance_round()
        if f.terminal:
            print(f"  seed={seed} round={warm}: 已终局，跳过")
            continue
        m = FlatRolloutGreedy(net_fn, k=300, rollouts_per_candidate=1, horizon=16,
                              max_total_rounds=10 ** 9, value_model=vm, feature_extractor=feat,
                              rollout_mode="idle")
        m.search(f.clone(), 0)
        bundles = list(m.last_root.bundles)
        priors = np.asarray([c.prior for c in m.last_root.children], dtype=float)
        values = np.asarray(m.last_scores, dtype=float)
        n = len(values)
        if n < 10:
            print(f"  seed={seed} round={warm}: 候选太少({n})，跳过")
            continue
        rho = spearman(priors, values)
        best = int(np.argmax(values))
        best_v = values[best]
        order_prior = np.argsort(-priors)
        rank_of_best = int(np.where(order_prior == best)[0][0]) + 1
        # recall@k：只用先验 top-k，最优动作是否还在里面
        recalls = {}
        for k in (1, 4, 8, 16, 32, 64):
            topk = order_prior[:k]
            recalls[k] = bool(best in topk)
        v_top1 = values[order_prior[0]]
        print(f"\n  seed={seed} round={warm}: 候选 {n} 个")
        print(f"    先验 vs 估值 Spearman rho = {rho:+.3f}")
        print(f"    估值最优动作的先验排名 = {rank_of_best}/{n}")
        for k, hit in recalls.items():
            print(f"    recall@{k:<3} = {'✅' if hit else '❌'}")
        print(f"    先验 top-1 的估值 {v_top1:+.4f}   vs   最优估值 {best_v:+.4f}   "
              f"（差 {best_v - v_top1:+.4f}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
