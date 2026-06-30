"""Per-round profiling: time breakdown of each agent's decision per round."""
from __future__ import annotations
import sys, time, csv
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND


def _time_agent(agent_type: str, seed: int = 42, max_rnd: int = 50) -> list[dict]:
    """Run first `max_rnd` rounds, recording per-round timing & illegals."""
    import time

    if agent_type == "example":
        from AI.ai_example import AI as AI0
        from AI.ai_example import AI as AI1
        a0, a1 = AI0(seed=seed), AI1(seed=seed + 10000)
    elif agent_type == "neural":
        from my_ai.agent import NeuralAgent
        from my_ai.network import create_zero_model
        a0 = NeuralAgent(model=create_zero_model())
        a1 = NeuralAgent(model=create_zero_model())
    elif agent_type == "neural_vs_example":
        from AI.ai_example import AI as AI1
        from my_ai.agent import NeuralAgent
        from my_ai.network import create_zero_model
        a0 = NeuralAgent(model=create_zero_model())
        a1 = AI1(seed=seed)
    else:
        raise ValueError(agent_type)

    our_player = seed % 2
    p0_agent = a0 if our_player == 0 else a1
    p1_agent = a1 if our_player == 0 else a0

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    rounds_data = []

    for rnd in range(min(max_rnd, MAX_ROUND)):
        if state.terminal:
            break

        # Time player 0's decision
        t0 = time.perf_counter()
        ops0 = p0_agent.choose_operations(state, 0)
        t1 = time.perf_counter()

        # Time player 1's decision
        ops1 = p1_agent.choose_operations(state, 1)
        t2 = time.perf_counter()

        # Time resolve
        result = state.resolve_turn(ops0, ops1)
        t3 = time.perf_counter()

        rounds_data.append({
            "round": rnd,
            "p0_time_ms": (t1 - t0) * 1000,
            "p1_time_ms": (t2 - t1) * 1000,
            "resolve_ms": (t3 - t2) * 1000,
            "illegal0": len(result.illegal[0]),
            "illegal1": len(result.illegal[1]),
            "terminal": state.terminal,
        })

    return rounds_data


def _profile_three(agent_type: str, seeds: list[int], num_rounds: int = 50):
    """Profile 3 seeds for a given agent type, print summary."""
    import numpy as np
    all_rows = []
    for seed in seeds:
        data = _time_agent(agent_type, seed, num_rounds)
        for r in data:
            r["seed"] = seed
            r["agent_type"] = agent_type
        all_rows.extend(data)

    p0 = [r["p0_time_ms"] for r in all_rows]
    p1 = [r["p1_time_ms"] for r in all_rows]
    res = [r["resolve_ms"] for r in all_rows]
    il0 = [r["illegal0"] for r in all_rows]
    il1 = [r["illegal1"] for r in all_rows]

    print(f"\n{'='*65}")
    print(f"  {agent_type}  ({len(all_rows)} rounds across {len(seeds)} seeds)")
    print(f"{'='*65}")
    print(f"  P0 decision  : {np.mean(p0):6.1f}ms avg  [{np.min(p0):.0f}~{np.max(p0):.0f}]")
    print(f"  P1 decision  : {np.mean(p1):6.1f}ms avg  [{np.min(p1):.0f}~{np.max(p1):.0f}]")
    print(f"  Resolve      : {np.mean(res):6.1f}ms avg  [{np.min(res):.0f}~{np.max(res):.0f}]")
    print(f"  Total/round  : {np.mean(p0)+np.mean(p1)+np.mean(res):6.1f}ms")
    print(f"  Illegal/round: P0={np.mean(il0):.2f}  P1={np.mean(il1):.2f}")

    # Also show first 3 rounds from first seed for a detailed look
    first_seed = seeds[0]
    first_data = [r for r in all_rows if r["seed"] == first_seed][:3]
    print(f"\n  Sample (seed={first_seed}, first 3 rounds):")
    print(f"  {'rnd':>4}  {'p0(ms)':>8}  {'p1(ms)':>8}  {'resolve':>8}  {'il0':>4}  {'il1':>4}")
    for d in first_data:
        print(f"  {d['round']:4d}  {d['p0_time_ms']:8.1f}  {d['p1_time_ms']:8.1f}  "
              f"{d['resolve_ms']:8.1f}  {d['illegal0']:4d}  {d['illegal1']:4d}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    seeds = [42, 43, 44]  # 3 different seeds
    _profile_three("example", seeds, args.rounds)
    _profile_three("neural", seeds, args.rounds)
    _profile_three("neural_vs_example", seeds, args.rounds)
