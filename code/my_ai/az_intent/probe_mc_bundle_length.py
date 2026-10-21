"""MC 菜单里有"动作组"吗？—— 实测 `_expand` 产出的 bundle 长度分布。

结论预期：
  · MC 菜单（`_cands`）= 单个 op（类号 / (类,x,y) / ('op',...)）
  · 但 pinned = [类]*3 ⇒ **三个头都被钉成同一个类**，各自选位置
  · 所以最终 action group 可能是 1–3 个同类 op（例如 3 次建塔）
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt("checkpoints/posnet_A_k5_m32.pt", feat,
                                  value_tanh=True, value_rel_to_abs=0.0)

searcher = BundleMCTS(net_fn, iterations=1, max_depth_rounds=4, k=24, sample_mult=15,
                      t_class=0.5, t_pos=1.0, c_puct=1.25, seed=1,
                      search_mode="pos-only", pos_pin="mc_quiet",
                      mc_every=5, mc_horizon=64, mc_root_only=1,
                      skip_single_candidate=True)

lengths = collections.Counter()
classes = collections.Counter()
examples = {}
for round_index in (41, 42, 43, 44, 56, 57, 58, 59, 71, 72, 73, 74, 86, 87, 88, 89, 101, 102, 103, 104):
    facade = GameStateFacade.initial(732)
    for _ in range(round_index):
        if facade.terminal:
            break
        facade.advance_round()
    if facade.terminal:
        continue
    for player in (0, 1):
        root = BundleNode(state=facade.clone(), player=player)
        searcher.last_root = root
        searcher._mc_root = root
        try:
            searcher._expand(root, 24)
        except Exception as exc:  # noqa: BLE001
            print(f"r={round_index} p={player} 异常 {type(exc).__name__}: {exc}")
            continue
        arbitrated = (round_index % 5 == 0)
        for b in root.bundles:
            lengths[len(b)] += 1
            for op in b:
                classes[op[0]] += 1
        key = (round_index % 5 == 0, len(root.bundles))
        if key not in examples and root.bundles:
            examples[key] = root.bundles[:3]
        print(f"r={round_index:>3} p={player} 仲裁={arbitrated} 候选={len(root.bundles):>2} "
              f"长度分布={dict(collections.Counter(len(b) for b in root.bundles))} "
              f"例={root.bundles[0] if root.bundles else None}")

print(f"\n=== 全体 bundle 长度分布 ===")
total = sum(lengths.values())
for n in sorted(lengths):
    print(f"  {n} 个 op: {lengths[n]:>5} 条 ({100*lengths[n]/total:.1f}%)")
print(f"\n=== op 类型分布 ===")
for cid, cnt in classes.most_common():
    print(f"  类 {cid}: {cnt}")
