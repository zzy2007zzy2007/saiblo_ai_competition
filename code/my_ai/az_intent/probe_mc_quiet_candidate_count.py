"""在 M6c-ROOT 口径（pos-only + pos_pin=mc_quiet）下，每次决策到底有几个候选？

动机：文档里"pos-only ⇒ ~97% 回合只有 1 个候选"是在 **pos_pin=argmax** 下测的
（类头塌缩 ⇒ argmax 类常常不可执行）。但 M6c-ROOT 用的是 **pos_pin=mc_quiet**：
类由 MC 仲裁选出、且是**可执行**的，于是同类的合法位置可能很多 ⇒ 候选数应该远不止 1。
这决定了"位置搜索"到底有多重要。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for p in [REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"]:
    sys.path.insert(0, str(p))

from SDK.utils.features import FeatureExtractor  # noqa: E402
from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt  # noqa: E402
from my_ai.az_intent.bundle_mcts import BundleMCTS, BundleNode  # noqa: E402
from my_ai.az_intent.game_state_facade import GameStateFacade  # noqa: E402

CKPT = "checkpoints/posnet_A_k5_m32.pt"
feat = FeatureExtractor(max_actions=96)
_, net_fn = make_net_fn_from_ckpt(CKPT, feat, value_tanh=True, value_rel_to_abs=0.0)

# 与 M6c-ROOT 一致；只把 MC 的"贵"部分保留（mc_every=5 与部署同）
searcher = BundleMCTS(net_fn, iterations=1, max_depth_rounds=4, k=24, sample_mult=15,
                      t_class=0.5, t_pos=1.0, c_puct=1.25, seed=1,
                      search_mode="pos-only", pos_pin="mc_quiet",
                      mc_every=5, mc_horizon=256, mc_root_only=1,
                      skip_single_candidate=True)

counts = []
t_total = 0.0
for round_index in (60, 100, 140, 180, 220, 260, 300, 340, 380, 420):
    facade = GameStateFacade.initial(732)
    for _ in range(round_index):
        if facade.terminal:
            break
        facade.advance_round()
    if facade.terminal:
        print(f"r={round_index}: 已终局，跳过")
        continue
    for player in (0, 1):
        root = BundleNode(state=facade.clone(), player=player)
        searcher.last_root = root
        searcher._mc_root = root          # ← 与部署一致：MC 仲裁只在根生效
        t0 = time.perf_counter()
        searcher._expand(root, 24)
        t_total += time.perf_counter() - t0
        n = len(root.bundles)
        if n > 0:
            counts.append(n)
        print(f"r={round_index:>3} p={player}: 候选 {n:>2} 个  "
              f"（前3: {[b for b in root.bundles[:3]]}）")

counts = np.asarray(counts)
print(f"\n候选数：中位 {np.median(counts):.0f}  均值 {counts.mean():.1f}  "
      f"范围 {counts.min()}–{counts.max()}")
print(f"只有 1 个候选的比例：{100.0 * (counts <= 1).mean():.0f}%")
print(f"每次 _expand 平均耗时 {t_total / max(len(counts),1)*1000:.0f} ms（含 MC 仲裁，16 线程）")
