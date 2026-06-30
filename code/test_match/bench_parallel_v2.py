"""Diagnose: is PyTorch MKL thread contention the issue in multiprocessing?"""
from __future__ import annotations
import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ── Worker with explicit torch thread control ──────────────────────

def _worker_single_thread(agent_type: str, seed: int) -> float:
    """Run one match with torch.num_threads=1."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    import torch
    torch.set_num_threads(1)

    from my_ai.agent import NeuralAgent
    from my_ai.network import create_zero_model

    if agent_type == "neural":
        a0 = NeuralAgent(model=create_zero_model())
        a1 = NeuralAgent(model=create_zero_model())
    elif agent_type == "neural_vs_example":
        from AI.ai_example import AI as ExampleAI
        a0 = NeuralAgent(model=create_zero_model())
        a1 = ExampleAI(seed=seed)
    else:
        raise ValueError(agent_type)

    our_p = seed % 2
    p0 = a0 if our_p == 0 else a1
    p1 = a1 if our_p == 0 else a0

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        state.resolve_turn(p0.choose_operations(state, 0), p1.choose_operations(state, 1))
    hp0, hp1 = state.bases[0].hp, state.bases[1].hp
    if hp0 <= 0 and hp1 <= 0:
        return 0.5
    return 1.0 if hp0 > hp1 else 0.0


# ── Same but WITHOUT torch thread control (baseline) ────────────────

def _worker_default_thread(agent_type: str, seed: int) -> float:
    """Run one match WITHOUT setting torch.num_threads."""
    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from my_ai.agent import NeuralAgent
    from my_ai.network import create_zero_model

    if agent_type == "neural":
        a0 = NeuralAgent(model=create_zero_model())
        a1 = NeuralAgent(model=create_zero_model())
    elif agent_type == "neural_vs_example":
        from AI.ai_example import AI as ExampleAI
        a0 = NeuralAgent(model=create_zero_model())
        a1 = ExampleAI(seed=seed)

    our_p = seed % 2
    p0 = a0 if our_p == 0 else a1
    p1 = a1 if our_p == 0 else a0

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        state.resolve_turn(p0.choose_operations(state, 0), p1.choose_operations(state, 1))
    hp0, hp1 = state.bases[0].hp, state.bases[1].hp
    if hp0 <= 0 and hp1 <= 0:
        return 0.5
    return 1.0 if hp0 > hp1 else 0.0


def bench(label: str, fn, agent_type: str, n: int, workers: int):
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.starmap(fn, [(agent_type, s) for s in range(n)])
    dt = time.perf_counter() - t0
    import numpy as np
    wr = np.mean(results)
    per_game = dt / n * 1000
    gps = n / dt
    print(f"  {label:>30s}: {n:2d} games  {dt:7.1f}s  {per_game:5.0f}ms/g  {gps:4.1f}g/s  wr={wr:.3f}")


if __name__ == "__main__":
    N, W = 24, 12
    print(f"Compare: torch.num_threads=1  vs  default ({W} workers, {N} games each)")
    print("=" * 80)
    # With single-threaded PyTorch
    bench("Neural vs Neural (1-thread)", _worker_single_thread, "neural", N, W)
    bench("Neural vs Example (1-thread)", _worker_single_thread, "neural_vs_example", N, W)
    print("---")
    # With default (multi-threaded) PyTorch
    bench("Neural vs Neural (default)", _worker_default_thread, "neural", N, W)
    bench("Neural vs Example (default)", _worker_default_thread, "neural_vs_example", N, W)
