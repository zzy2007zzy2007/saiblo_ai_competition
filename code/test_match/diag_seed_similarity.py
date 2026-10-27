"""两个补充量（2026-10-26 夜）：

A) 不同 FLAT_SEED 的"轨迹相似度"：种子 53 vs 61 生成的**去重叶子轨迹集合**的 Jaccard 重叠率。
   · 重叠率低 ⇒ 它们是两套完全不同的"对手性格"（解释为什么胜率差 15-20pp）
B) act_prob 0.3 / 0.6 / 1.0 下，64 个分支的**去重叶子数**能到多少（有效采样能不能从 ~18 提到 50+）。

依赖：`flat_rollout_greedy.py` 里 `_REC_SAMPLES=1` 时会记录
  `last_tree_stats = (节点数, 叶子数, 去重叶子数)` 与 `last_leaf_tokens`（本脚本要求后者）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import statistics

import torch

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor


def build(feat, net_fn, vm, seed: int, act_prob: float) -> FlatRolloutGreedy:
    return FlatRolloutGreedy(
        net_fn, k=48, rollouts_per_candidate=64, horizon=16, max_total_rounds=10 ** 9,
        value_model=vm, feature_extractor=feat,
        rollout_mode="random", rollout_random_side="opponent",
        rollout_prob_mode="decay", rollout_decay_c=0.3, rollout_decay_k=1.0,
        rollout_act_prob=act_prob, rollout_seed=seed, rollout_share_prefix=True,
    )


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(str(REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                      value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(str(REPO / "checkpoints/posnet_A_k5_m32.pt"))
    vm.eval()
    states = []
    for sd in (10, 11, 12):
        st = GameStateFacade.initial(seed=sd, cold_handle_rule_illegal=True)
        for _ in range(60):
            st.advance_round()
        states.append(st.clone())

    # ── B) (C, k) 扫描：这两个才是衰减模式真正生效的旋钮 ─────────────────
    print("=== B) (DECAY_C, DECAY_K) 对去重叶子数的影响（N=64, k=48）===")
    for c, k in ((0.3, 1.0), (1.0, 1.0), (1.0, 3.0), (0.6, 5.0)):
        s = FlatRolloutGreedy(
            net_fn, k=48, rollouts_per_candidate=64, horizon=16, max_total_rounds=10 ** 9,
            value_model=vm, feature_extractor=feat,
            rollout_mode="random", rollout_random_side="opponent",
            rollout_prob_mode="decay", rollout_decay_c=c, rollout_decay_k=k,
            rollout_act_prob=0.3, rollout_seed=53, rollout_share_prefix=True)
        leaves = []
        for st in states:
            s.search(st, 0)
            if getattr(s, "last_tree_stats", None):
                leaves.append(s.last_tree_stats[2])
        exp_acts = c * sum(__import__("math").exp(-r / k) for r in range(16))
        if leaves:
            print(f"  C={c:<4} k={k:<4} 期望动作 {exp_acts:.2f}/rollout  去重叶子={leaves}  平均 {statistics.mean(leaves):.1f}/64")

    # ── A) 种子 53 vs 61 的轨迹重叠 ─────────────────────────────────────
    print("\n=== A) FLAT_SEED 53 vs 61 的轨迹集合重叠（Jaccard）===")
    sets53, sets61 = [], []
    for st in states:
        cur = []
        for seed, bucket in ((53, sets53), (61, sets61)):
            s = build(feat, net_fn, vm, seed, 0.3)
            s.search(st, 0)
            toks = set(getattr(s, "last_leaf_tokens", []) or [])
            bucket.append(toks)
            cur.append(toks)
        if len(cur) == 2:
            a, b = cur
            inter = len(a & b)
            union = len(a | b) or 1
            print(f"  局面: |53|={len(a)}  |61|={len(b)}  交集={inter}  "
                  f"Jaccard={100*inter/union:.1f}%")
    if sets53 and sets61:
        all53 = set().union(*sets53)
        all61 = set().union(*sets61)
        print(f"  合并: |53|={len(all53)}  |61|={len(all61)}  交集={len(all53 & all61)}  "
              f"Jaccard={100*len(all53 & all61)/max(1,len(all53 | all61)):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
