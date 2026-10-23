"""诊断：随机化 rollout 里双方到底随机出了什么动作（2026-10-23 用户要求）。

做法：拿一个真实中局，手动跑 3 个样本 × 16 回合，逐回合记录：
  每方随机到的 op（建塔/降级）、执行前的金币、是否真的执行成功、执行后塔数。
最后汇总：每个样本建了几次塔、拆了几次、金币净变化、塔数变化。
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
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType, HIGHLAND_CELLS

HORIZON = 16
ACT_PROB = 0.5
BUILD_RATIO = 0.7


def main() -> int:
    f = GameStateFacade.initial(seed=1230, cold_handle_rule_illegal=True)
    for _ in range(140):
        f.advance_round()
    print(f"局面：round={f.round_index} 金币={list(f.coins)} 塔数={[len(list(f.towers_of(p))) for p in (0,1)]}")

    # 借用 FlatRolloutGreedy 的 _random_actions（不跑搜索，只借这个方法）
    m = FlatRolloutGreedy.__new__(FlatRolloutGreedy)   # 只借用 _random_actions，不构造完整搜索器
    m.rollout_act_prob = ACT_PROB
    m.rollout_build_ratio = BUILD_RATIO
    m.rollout_random_side = "both"
    ctx = {"cells": {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}}

    for sample in range(3):
        sim = f.clone()
        rng = np.random.default_rng(1234 + sample)
        n_build = n_down = n_fail = 0
        coins0 = list(sim.coins)
        tw0 = [len(list(sim.towers_of(p))) for p in (0, 1)]
        print(f"\n--- 样本 {sample} ---")
        for rnd in range(HORIZON):
            if sim.terminal:
                break
            before = {p: (int(sim.coins[p]), len(list(sim.towers_of(p)))) for p in (0, 1)}
            # 记录这一回合内每方做了什么：包一层，读取 apply 前后的差异
            m._random_actions(sim, rng, ctx, root_player=None)
            sim.advance_round()
            after = {p: (int(sim.coins[p]), len(list(sim.towers_of(p)))) for p in (0, 1)}
            line = []
            for p in (0, 1):
                dc = after[p][0] - before[p][0]
                dt = after[p][1] - before[p][1]
                if dc != 0 or dt != 0:
                    line.append(f"P{p}: 金币{dc:+d} 塔{dt:+d}")
            if line:
                print(f"  r{rnd:>2}: " + "  ".join(line))
        coins1 = list(sim.coins)
        tw1 = [len(list(sim.towers_of(p))) for p in (0, 1)]
        print(f"  合计：金币 {[coins0[p] for p in (0,1)]} → {coins1}   塔数 {tw0} → {tw1}"
              f"   terminal={sim.terminal} hp={[int(b.hp) for b in sim.bases]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
