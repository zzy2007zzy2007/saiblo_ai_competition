#!/usr/bin/env python3
"""Profile where time is spent in a match to identify bottlenecks."""

import time
import sys
from pathlib import Path

REPO_ROOT = Path("d:/2026智能体大赛_新2/Ant-Game").resolve()
sys.path.insert(0, str(REPO_ROOT))

from AI.ai_greedy import AI as GreedyAI, _to_greedy_info, _to_sdk_operation
from AI.ai_random import AI as RandomAI
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND


def profile_match(seed: int = 7, max_rounds: int = 100):
    state = GameState.initial(seed=seed, movement_policy="enhanced")
    agent0 = GreedyAI()
    agent1 = RandomAI(seed=seed)
    if hasattr(agent1, "on_match_start"):
        agent1.on_match_start(1, seed)

    timers = {
        "ai_decision": 0.0,
        "state_conversion": 0.0,
        "resolve_turn": 0.0,
        "public_state": 0.0,
    }

    for round_idx in range(min(MAX_ROUND, max_rounds)):
        if state.terminal:
            break

        # AI0 decision
        t0 = time.perf_counter()
        t_conv = time.perf_counter()
        greedy_info = _to_greedy_info(state)
        timers["state_conversion"] += time.perf_counter() - t_conv
        raw_ops = agent0(0, greedy_info)
        ops0 = [_to_sdk_operation(op) for op in raw_ops]
        timers["ai_decision"] += time.perf_counter() - t0

        # AI1 decision
        t0 = time.perf_counter()
        ops1 = agent1.choose_operations(state, 1)
        timers["ai_decision"] += time.perf_counter() - t0

        # Resolve turn
        t0 = time.perf_counter()
        resolution = state.resolve_turn(ops0, ops1)
        timers["resolve_turn"] += time.perf_counter() - t0

        # Public state
        t0 = time.perf_counter()
        public_state = state.to_public_round_state()
        timers["public_state"] += time.perf_counter() - t0

        if hasattr(agent1, "on_round_state"):
            agent1.on_round_state(public_state)

        if resolution.terminal:
            break

    total = sum(timers.values())
    print(f"对战 {round_idx + 1} 回合性能分析:")
    print(f"  AI 决策(含状态转换): {timers['ai_decision']:.3f}s ({timers['ai_decision']/total*100:.1f}%)")
    print(f"    其中状态转换:       {timers['state_conversion']:.3f}s")
    print(f"  引擎结算(resolve):     {timers['resolve_turn']:.3f}s ({timers['resolve_turn']/total*100:.1f}%)")
    print(f"  局面序列化(public):    {timers['public_state']:.3f}s ({timers['public_state']/total*100:.1f}%)")
    print(f"  总耗时:               {total:.3f}s")
    print(f"  平均每回合:           {total/(round_idx+1)*1000:.1f}ms")
    return timers


if __name__ == "__main__":
    profile_match(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 7, max_rounds=120)
