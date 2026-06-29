#!/usr/bin/env python3
"""Run a full match between two AIs using the pure Python backend."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path("d:/2026智能体大赛_新2/Ant-Game").resolve()
sys.path.insert(0, str(REPO_ROOT))

from AI.ai_greedy import AI as GreedyAI, _to_greedy_info, _to_sdk_operation
from AI.ai_random import AI as RandomAI
from AI.ai_example import AI as ExampleAI
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND


def run_match(ai0_cls, ai1_cls, seed: int = 7, ai0_kwargs=None, ai1_kwargs=None):
    state = GameState.initial(seed=seed, movement_policy="enhanced")

    kwargs0 = ai0_kwargs or {}
    kwargs1 = ai1_kwargs or {}
    agent0 = ai0_cls(**kwargs0)
    agent1 = ai1_cls(**kwargs1)

    if hasattr(agent0, "on_match_start"):
        agent0.on_match_start(0, seed)
    if hasattr(agent1, "on_match_start"):
        agent1.on_match_start(1, seed)

    round_idx = 0
    for round_idx in range(MAX_ROUND):
        if state.terminal:
            break

        if hasattr(agent0, "choose_operations"):
            ops0 = agent0.choose_operations(state, 0)
        elif callable(agent0):
            ops0 = [_to_sdk_operation(op) for op in agent0(0, _to_greedy_info(state))]
        else:
            raise TypeError("agent0 has no supported interface")

        if hasattr(agent1, "choose_operations"):
            ops1 = agent1.choose_operations(state, 1)
        elif callable(agent1):
            ops1 = [_to_sdk_operation(op) for op in agent1(1, _to_greedy_info(state))]
        else:
            raise TypeError("agent1 has no supported interface")

        resolution = state.resolve_turn(ops0, ops1)

        public_state = state.to_public_round_state()
        if hasattr(agent0, "on_round_state"):
            agent0.on_round_state(public_state)
        if hasattr(agent1, "on_round_state"):
            agent1.on_round_state(public_state)

        if resolution.terminal:
            break

    return {
        "rounds": round_idx + 1,
        "terminal": state.terminal,
        "winner": state.winner,
        "hp": [state.bases[0].hp, state.bases[1].hp],
        "coins": list(state.coins),
        "tower_count": [
            len([t for t in state.towers if t.player == 0]),
            len([t for t in state.towers if t.player == 1]),
        ],
        "ant_count": len(state.ants),
        "die_count": list(state.die_count),
    }


def print_summary(label: str, result: dict):
    print(f"\n=== {label} ===")
    print(f"  回合: {result['rounds']}  |  胜者: P{result['winner']}  |  "
          f"基地: P0={result['hp'][0]}, P1={result['hp'][1]}  |  "
          f"金币: {result['coins']}  |  塔: {result['tower_count']}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7

    for label, match in [
        ("Greedy vs Random", lambda: run_match(GreedyAI, RandomAI, seed=seed)),
        ("Random vs Random", lambda: run_match(RandomAI, RandomAI, seed=seed)),
        ("Greedy vs Greedy", lambda: run_match(GreedyAI, GreedyAI, seed=seed)),
        ("Example vs Random", lambda: run_match(ExampleAI, RandomAI, seed=seed, ai0_kwargs={"seed": seed}, ai1_kwargs={"seed": seed})),
    ]:
        print_summary(label, match())
