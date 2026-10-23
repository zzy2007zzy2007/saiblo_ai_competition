"""量化：随机 rollout 的 N 个样本里，到底有几个**不同的轨迹/叶子**（= 记忆化能省多少）。

动机（用户质疑）：衰减模式下 p(r) 很快变小，很多样本可能**一次动作都没抽到** ⇒
轨迹与"静止基线"完全相同 ⇒ 缓存命中率高，N 次采样实际只提供了很少的新信息。

输出：对每种配置，跑 N=32 个样本，统计
  · 不同的动作序列数
  · 不同的叶子状态数（按 coins/towers/hp/ants 摘要哈希）
  · 与"静止"叶子相同的样本数
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(r"D:\2026智能体大赛_新2_distill")
for _p in (REPO / "code", REPO / "code" / "cpp_engine", REPO / "Ant-Game", REPO / "Ant-Game" / "SDK"):
    sys.path.insert(0, str(_p))

import numpy as np

from my_ai.az_intent.game_state_facade import GameStateFacade
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy
from SDK.utils.constants import HIGHLAND_CELLS

N = 32
H = 16


def leaf_key(sim) -> str:
    parts = [
        str([int(c) for c in sim.coins]),
        str([int(b.hp) for b in sim.bases]),
        str(sorted((int(t.tower_id), int(t.x), int(t.y), int(t.tower_type)) for t in sim.towers)),
        str(sorted((int(a.ant_id), int(a.x), int(a.y), int(a.hp)) for a in sim.ants)),
    ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def run(mode: str, side: str, prob_mode: str, c: float, k: float, prob: float):
    f = GameStateFacade.initial(seed=1460, cold_handle_rule_illegal=True)
    for _ in range(140):
        f.advance_round()
    # 候选 op：建一座塔（固定），模拟"打完候选动作之后"的 rollout
    root = f.clone()
    ctx = {"cells": {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}}
    m = FlatRolloutGreedy.__new__(FlatRolloutGreedy)
    m.rollout_act_prob = prob
    m.rollout_build_ratio = 0.7
    m.rollout_random_side = side
    m.rollout_prob_mode = prob_mode
    m.rollout_decay_c = c
    m.rollout_decay_k = k

    keys, seqs = [], []
    for s in range(N):
        sim = root.clone()
        rng = np.random.default_rng(9_000_000 + s)   # 与线上同构：只由种子决定
        seq = []
        for r in range(H):
            if sim.terminal:
                break
            before = [int(x) for x in sim.coins]
            m._random_actions(sim, rng, ctx, None, r)
            after = [int(x) for x in sim.coins]
            if before != after:
                seq.append(r)
            sim.advance_round()
        keys.append(leaf_key(sim))
        seqs.append(tuple(seq))

    # 静止基线
    idle = root.clone()
    for _ in range(H):
        if idle.terminal:
            break
        idle.advance_round()
    idle_key = leaf_key(idle)

    print(f"  {mode:<26} side={side:<8} 动作序列 {len(set(seqs)):>2}/{N} 个不同，"
          f"叶子 {len(set(keys)):>2}/{N} 个不同，与静止相同 {sum(1 for x in keys if x == idle_key):>2}/{N}")
    return len(set(keys)), sum(1 for x in keys if x == idle_key)


print(f"样本数 N={N}，视野 H={H}")
run("const p=0.5", "both", "const", 0.0, 3.0, 0.5)
run("const p=0.5", "opponent", "const", 0.0, 3.0, 0.5)
run("decay C=.6 k=3", "opponent", "decay", 0.6, 3.0, 0.6)
run("decay C=.6 k=3", "both", "decay", 0.6, 3.0, 0.6)
run("decay C=.3 k=1", "opponent", "decay", 0.3, 1.0, 0.3)
run("first only C=.6", "opponent", "first", 0.6, 3.0, 0.6)
