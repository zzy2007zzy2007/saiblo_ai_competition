"""Benchmark: ES-trainer-style multiprocessing match throughput.

Mimics es_train.py _eval_worker exactly:
  - cold_handle_rule_illegal=True
  - seed parity controls first/second player
  - full MAX_ROUND loop with resolve_turn
  - NeuralAgent with zero-weight model
"""
from __future__ import annotations

import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np


def _worker(agent_type: str, seed: int) -> float:
    """Run one match, return 1.0/0.5/0.0."""
    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    if agent_type == "example":
        from AI.ai_example import AI as ExampleAI
        a0 = ExampleAI(seed=seed)
        a1 = ExampleAI(seed=seed + 10000)
    elif agent_type == "neural":
        from my_ai.agent import NeuralAgent
        from my_ai.network import create_zero_model
        a0 = NeuralAgent(model=create_zero_model())
        a1 = NeuralAgent(model=create_zero_model())
    elif agent_type == "neural_vs_example":
        from AI.ai_example import AI as ExampleAI
        from my_ai.agent import NeuralAgent
        from my_ai.network import create_zero_model
        a0 = NeuralAgent(model=create_zero_model())
        a1 = ExampleAI(seed=seed)
    else:
        raise ValueError(f"unknown agent_type: {agent_type}")

    # Alternate first/second player by seed parity (same as ES trainer)
    our_player = seed % 2
    opp_player = 1 - our_player
    p0_agent = a0 if our_player == 0 else a1
    p1_agent = a1 if our_player == 0 else a0

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        ops0 = p0_agent.choose_operations(state, 0)
        ops1 = p1_agent.choose_operations(state, 1)
        state.resolve_turn(ops0, ops1)

    hp0, hp1 = state.bases[0].hp, state.bases[1].hp
    if hp0 <= 0 and hp1 <= 0:
        return 0.5
    return 1.0 if hp0 > hp1 else 0.0


def bench(label: str, agent_type: str, n: int, workers: int):
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.starmap(_worker, [(agent_type, s) for s in range(n)])
    dt = time.perf_counter() - t0
    wr = np.mean(results)
    per_game = dt / n * 1000
    gps = n / dt
    print(f"{label:>28s}: {n:2d} games  {dt:7.1f}s  "
          f"{per_game:5.0f}ms/g  {gps:5.1f}g/s  win_rate={wr:.3f}")


if __name__ == "__main__":
    from SDK.utils.constants import MAX_ROUND
    N = 48
    W = 12
    print(f"Benchmark: {N} games each, {W} workers, cold_handle=True, max_round={MAX_ROUND}")
    print("=" * 80)
    bench("Example vs Example", "example", N, W)
    bench("Neural vs Neural", "neural", N, W)
    bench("Neural vs Example", "neural_vs_example", N, W)
