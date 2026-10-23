"""量化"前缀共享"：随机 rollout 的 N 条轨迹建成前缀树后，实际要模拟多少回合。

用户的关键观察（2026-10-23）：叶子相同概率不大，但**中间局面相同的不少**——
很多样本在前几回合什么都没抽到 ⇒ 共享同一段前缀 ⇒ 如果按树共享状态，就能跳过重复模拟。

做法：每条轨迹记录逐回合的"动作事件 token"（谁做了什么），建成前缀树；
  · 唯一节点数 = 记忆化后真正需要模拟的回合数
  · 节省率 = 1 − 唯一节点数 / (N × H)
同时报告"首个随机动作出现的回合"的分布（前缀长度的直接度量）。
"""
from __future__ import annotations

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


def trajectory_tokens(m, root, ctx, seed, side, prob_mode, c, k, prob):
    """返回逐回合的 token 列表（'S' = 该回合无动作）。"""
    m.rollout_act_prob = prob
    m.rollout_build_ratio = 0.7
    m.rollout_random_side = side
    m.rollout_prob_mode = prob_mode
    m.rollout_decay_c = c
    m.rollout_decay_k = k
    sim = root.clone()
    rng = np.random.default_rng(seed)
    toks = []
    for r in range(H):
        if sim.terminal:
            toks.append("T")
            break
        before = ([int(x) for x in sim.coins],
                  sorted((int(t.tower_id), int(t.tower_type)) for t in sim.towers))
        m._random_actions(sim, rng, ctx, None, r)
        after = ([int(x) for x in sim.coins],
                 sorted((int(t.tower_id), int(t.tower_type)) for t in sim.towers))
        toks.append("X" if before != after else "S")   # X=动了，S=没动
        sim.advance_round()
    return toks


def trie_nodes(trajs):
    """前缀树的节点数（= 记忆化后需要模拟的回合数）。"""
    root = {}
    nodes = 0
    for t in trajs:
        cur = root
        for tok in t:
            if tok not in cur:
                cur[tok] = {}
                nodes += 1
            cur = cur[tok]
    return nodes


def measure(label, side, prob_mode, c, k, prob):
    f = GameStateFacade.initial(seed=1460, cold_handle_rule_illegal=True)
    for _ in range(140):
        f.advance_round()
    root = f.clone()
    ctx = {"cells": {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}}
    m = FlatRolloutGreedy.__new__(FlatRolloutGreedy)
    trajs = [trajectory_tokens(m, root, ctx, 9_000_000 + s, side, prob_mode, c, k, prob)
             for s in range(N)]
    nodes = trie_nodes(trajs)
    total = N * H
    # 首个"动"的回合分布
    firsts = []
    for t in trajs:
        idx = next((i for i, x in enumerate(t) if x == "X"), None)
        firsts.append(H if idx is None else idx)
    print(f"  {label:<26} 唯一节点 {nodes:>3}/{total} = 省 {100*(1-nodes/total):>4.0f}%"
          f"   首个随机动作回合: 中位 {int(np.median(firsts))}  全静止样本 {sum(1 for x in firsts if x == H)}/{N}")


print(f"N={N} 条轨迹 × H={H} 回合 = {N*H} 个回合")
measure("const p=0.5 both", "both", "const", 0.0, 3.0, 0.5)
measure("decay C=.6 k=3 opponent", "opponent", "decay", 0.6, 3.0, 0.6)
measure("decay C=.6 k=3 both", "both", "decay", 0.6, 3.0, 0.6)
measure("decay C=.3 k=1 opponent", "opponent", "decay", 0.3, 1.0, 0.3)
measure("first only C=.6", "opponent", "first", 0.6, 3.0, 0.6)
