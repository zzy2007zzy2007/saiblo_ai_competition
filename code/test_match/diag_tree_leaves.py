"""统计：N=64、前缀树模式下，不同 FLAT_SEED 的叶子节点数（2026-10-26）。

含义：N=64 个随机分支经前缀树合并后，真正**不同**的末状态有多少个。
  · 叶子数 ≈ 64 ⇒ 随机性充分（每条分支都不一样）
  · 叶子数 << 64 ⇒ 大量分支其实重复（等于有效采样变少）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import statistics
import torch

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor

SEEDS = [53, 61, 71, 83]


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

    print(f"  N=64，k=48，2 个局面 × 4 个 FLAT_SEED")
    for flat_seed in SEEDS:
        s = FlatRolloutGreedy(net_fn, k=48, rollouts_per_candidate=64, horizon=16,
                              max_total_rounds=10 ** 9, value_model=vm, feature_extractor=feat,
                              rollout_mode="random", rollout_random_side="opponent",
                              rollout_prob_mode="decay", rollout_decay_c=0.3, rollout_decay_k=1.0,
                              rollout_act_prob=0.3, rollout_seed=flat_seed,
                              rollout_share_prefix=True)
        leaves, nodes, uniq = [], [], []
        for st in states:
            s.search(st, 0)
            if getattr(s, "last_tree_stats", None):
                n, l, u = s.last_tree_stats
                nodes.append(n)
                leaves.append(l)
                uniq.append(u)
        if leaves:
            print(f"  FLAT_SEED={flat_seed:<3} 叶子={leaves} 去重叶子={uniq} 节点={nodes}  "
                  f"平均去重叶子 {statistics.mean(uniq):.1f}/64", flush=True)
        else:
            print(f"  FLAT_SEED={flat_seed:<3} 未记录到统计（检查是否走前缀树路径）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
