"""前缀树规模与耗时：对比旧配置 (C=.3,k=1) 与新配置 (C=1,k=3)（2026-10-26）。

关心：把随机化调强之后，**树节点数**会不会暴涨（决定 rollout 时间）。
  · 节点数 = 去重后的状态数（前缀树真正模拟过的局面）
  · 节点数 ↑ ⇒ clone/apply/advance 次数 ↑ ⇒ 时间 ↑
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

os.environ["AZAI_ROLLOUT_SAMPLES"] = "1"

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import torch

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor

CONFIGS = [(0.3, 1.0), (1.0, 1.0), (1.0, 3.0), (0.6, 5.0)]


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

    print("  N=64, k=48, 3 个局面（round 60）")
    print(f"  {'C':>4} {'k':>4} {'期望动作':>8} {'节点数':>10} {'去重叶子':>9} {'单次搜索 ms':>12}")
    for c, k in CONFIGS:
        s = FlatRolloutGreedy(
            net_fn, k=48, rollouts_per_candidate=64, horizon=16, max_total_rounds=10 ** 9,
            value_model=vm, feature_extractor=feat,
            rollout_mode="random", rollout_random_side="opponent",
            rollout_prob_mode="decay", rollout_decay_c=c, rollout_decay_k=k,
            rollout_act_prob=0.3, rollout_seed=53, rollout_share_prefix=True)
        nodes, leaves, times = [], [], []
        for st in states:
            t0 = time.perf_counter()
            s.search(st, 0)
            times.append((time.perf_counter() - t0) * 1000)
            if getattr(s, "last_tree_stats", None):
                nodes.append(s.last_tree_stats[0])
                leaves.append(s.last_tree_stats[2])
        exp_acts = c * sum(__import__("math").exp(-r / k) for r in range(16))
        if nodes:
            print(f"  {c:>4} {k:>4} {exp_acts:>8.2f} {statistics.mean(nodes):>10.0f} "
                  f"{statistics.mean(leaves):>9.1f} {statistics.mean(times):>12.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
