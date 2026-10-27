"""拆解 rollout 时间：树模拟 vs 值网前向（2026-10-27）。

做法：包装 `FlatRolloutGreedy._leaf_values` 计时（值网前向都从那里走），
与总耗时、`last_timing` 的 rollout 段对比，得到三者占比。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np
import torch

from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, load_three_models
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from my_ai.az_intent.game_state_facade import GameStateFacade
from SDK.utils.features import FeatureExtractor

_ACC = {"net_s": 0.0, "net_calls": 0, "net_leaves": 0}


def main() -> int:
    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt(str(REPO / "checkpoints/posnet_A_k5_m32.pt"), feat,
                                      value_tanh=True, value_rel_to_abs=0.0)
    _, _, vm = load_three_models(str(REPO / "checkpoints/posnet_A_k5_m32.pt"))
    vm.eval()
    print(f"  值网 device: {next(vm.parameters()).device}", flush=True)

    orig = FlatRolloutGreedy._leaf_values

    def timed(self, states, root_player):
        t0 = time.perf_counter()
        out = orig(self, states, root_player)
        _ACC["net_s"] += time.perf_counter() - t0
        _ACC["net_calls"] += 1
        _ACC["net_leaves"] += len(states)
        return out

    FlatRolloutGreedy._leaf_values = timed

    st = GameStateFacade.initial(seed=10, cold_handle_rule_illegal=True)
    for _ in range(60):
        st.advance_round()

    for tag, c, k, h in (("旧 C=.3 k=1 h=16", 0.3, 1.0, 16), ("新 C=1 k=3 h=8", 1.0, 3.0, 8)):
        s = FlatRolloutGreedy(net_fn, k=48, rollouts_per_candidate=64, horizon=h,
                              max_total_rounds=10 ** 9, value_model=vm, feature_extractor=feat,
                              rollout_mode="random", rollout_random_side="opponent",
                              rollout_prob_mode="decay", rollout_decay_c=c, rollout_decay_k=k,
                              rollout_act_prob=0.3, rollout_seed=53, rollout_share_prefix=True)
        _ACC.update({"net_s": 0.0, "net_calls": 0, "net_leaves": 0})
        t0 = time.perf_counter()
        s.search(st, 0)
        total = time.perf_counter() - t0
        print(f"\n  [{tag}] 总 {total*1000:.0f} ms", flush=True)
        print(f"    值网: {_ACC['net_s']*1000:6.0f} ms ({100*_ACC['net_s']/total:4.1f}%)  "
              f"调用 {_ACC['net_calls']} 次 / 叶子 {_ACC['net_leaves']} 个  "
              f"平均 {1000*_ACC['net_s']/max(1,_ACC['net_calls']):.1f} ms/次", flush=True)
        sim = total - _ACC["net_s"]
        print(f"    其余（树模拟+clone+apply+advance+杂项）: {sim*1000:6.0f} ms ({100*sim/total:4.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
