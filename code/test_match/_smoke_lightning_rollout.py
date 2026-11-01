# -*- coding: utf-8 -*-
"""冒烟：`AZAI_FLAT_ROLLOUT_MODE=lightning` 的 `_decide_actions` 语义（2026-11-02）。

直接构造 `FlatRolloutGreedy(rollout_mode="lightning", ...)`，用一个最小 fake sim 验证：
  1. `rollout_mode="lightning"` 能被构造（不再被 `__init__` 的 assert 拒绝）；
  2. 我方（root_player）在 cd==0 且金币≥90 时，随机动作 = `USE_LIGHTNING_STORM`（class 17 = op 21）；
  3. cd>0 或 金币<90 时**空过**（这一模式只有闪电一种动作，不建塔也不降级）；
  4. 侧 = 我方（root_player），不是对手（1-root_player）。

这是一个**只测 `_decide_actions` 分支**的最小单测；不跑完整搜索（value_model 不需要）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\2026智能体大赛_新2_distill")
for p in [ROOT / "code", ROOT / "Ant-Game"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SDK.utils.constants import HIGHLAND_CELLS, OperationType  # noqa: E402
from my_ai.az_intent.flat_rollout_greedy import FlatRolloutGreedy  # noqa: E402


class FakeSim:
    """最小 sim：`_decide_actions` 的 lightning 分支只读 coins / weapon_cooldowns。"""

    def __init__(self, coins0, coins1, cd0, cd1):
        self.coins = [coins0, coins1]
        self.weapon_cooldowns = np.zeros((2, 5), dtype=np.int16)
        self.weapon_cooldowns[0, 1] = cd0   # SuperWeaponType.LIGHTNING_STORM == 1
        self.weapon_cooldowns[1, 1] = cd1


def build(mode: str) -> FlatRolloutGreedy:
    return FlatRolloutGreedy(
        lambda s, p: None,
        k=48,
        rollouts_per_candidate=64,
        horizon=16,
        value_batch=32,
        eval_mode="hand",           # 不加载 value_model（本测不跑叶子评估）
        rollout_mode=mode,
        rollout_random_side="opponent",
        rollout_prob_mode="decay",
        rollout_decay_c=1.0,
        rollout_decay_k=3.0,
        rollout_share_prefix=True,
        rollout_seed=53,
        max_total_rounds=49152,
    )


def main() -> int:
    ctx = {"cells": {0: list(HIGHLAND_CELLS[0]), 1: list(HIGHLAND_CELLS[1])}}
    LIGHT = int(OperationType.USE_LIGHTNING_STORM)
    ok = True

    # 1) 构造通过
    searcher = build("lightning")
    assert searcher.rollout_mode == "lightning"
    print("[smoke] 1) FlatRolloutGreedy(rollout_mode='lightning') 构造通过 ✓")

    # 2) 我方 cd==0 + coins>=90 ⇒ 放闪电（侧 = 我方 0，不是对手 1）
    rng = np.random.default_rng(0)
    sim = FakeSim(coins0=120, coins1=0, cd0=0, cd1=0)
    ops = searcher._decide_actions(sim, rng, ctx, root_player=0, round_idx=0)
    print(f"[smoke] 2) root=0 coins0=120 cd0=0 -> {ops}")
    cond = bool(ops) and all(int(o[0]) == 0 and int(o[1]) == LIGHT for o in ops)
    print(f"         期望: 我方(player0) 放闪电 op21; 实际 {'✓' if cond else '✗'}")
    ok = ok and cond

    # 3) cd>0 ⇒ 空过
    rng2 = np.random.default_rng(1)
    ops2 = searcher._decide_actions(FakeSim(120, 0, 5, 0), rng2, ctx, root_player=0, round_idx=0)
    print(f"[smoke] 3) cd0=5 -> {ops2}  (期望空) {'✓' if not ops2 else '✗'}")
    ok = ok and not ops2

    # 4) coins<90 ⇒ 空过
    rng3 = np.random.default_rng(2)
    ops3 = searcher._decide_actions(FakeSim(30, 0, 0, 0), rng3, ctx, root_player=0, round_idx=0)
    print(f"[smoke] 4) coins0=30 -> {ops3}  (期望空) {'✓' if not ops3 else '✗'}")
    ok = ok and not ops3

    # 5) root_player=1 时侧 = 我方 1（不是对手 0）
    rng4 = np.random.default_rng(3)
    ops4 = searcher._decide_actions(FakeSim(0, 120, 0, 0), rng4, ctx, root_player=1, round_idx=0)
    cond4 = bool(ops4) and all(int(o[0]) == 1 and int(o[1]) == LIGHT for o in ops4)
    print(f"[smoke] 5) root=1 coins1=120 cd1=0 -> {ops4}  (期望 player1 放闪电) {'✓' if cond4 else '✗'}")
    ok = ok and cond4

    # 6) 落点在合法高地格（我方 HIGHLAND_CELLS）
    cond6 = all((int(o[2]), int(o[3])) in HIGHLAND_CELLS[0] for o in ops)
    print(f"[smoke] 6) 落点 ∈ HIGHLAND_CELLS[0] {'✓' if cond6 else '✗'}  -> {[(int(o[2]), int(o[3])) for o in ops]}")
    ok = ok and cond6

    print("\n" + ("✅ 全部通过" if ok else "❌ 有失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
