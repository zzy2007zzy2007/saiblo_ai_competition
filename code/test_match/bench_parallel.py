"""Quick test: verify multiprocessing Pool parallelism in current environment."""
from __future__ import annotations
import sys, time, multiprocessing as mp
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND


def _sleep_worker(sec: float) -> float:
    """Simple CPU-bound task to test parallelism."""
    import time
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sec:
        [x**2 for x in range(10000)]  # burn CPU
    return time.perf_counter() - t0


def _game_worker(seed: int) -> float:
    """Full game worker (same as ES trainer)."""
    import sys
    from pathlib import Path
    _RP = Path(__file__).resolve().parents[2] / "Ant-Game"
    _CODE = Path(__file__).resolve().parents[1]
    for p in (_RP, _CODE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from my_ai.agent import NeuralAgent
    from my_ai.network import create_zero_model
    from AI.ai_example import AI as ExampleAI
    
    a0 = NeuralAgent(model=create_zero_model())
    a1 = ExampleAI(seed=seed)
    our_p = seed % 2
    p0 = a0 if our_p == 0 else a1
    p1 = a1 if our_p == 0 else a0

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    t0 = time.perf_counter()
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        state.resolve_turn(p0.choose_operations(state, 0), p1.choose_operations(state, 1))
    return time.perf_counter() - t0


def test(worker_fn, label: str, n: int, workers: int):
    print(f"\n  {label}: {n} tasks, {workers} workers")
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        results = pool.map(worker_fn, range(n))
    dt = time.perf_counter() - t0
    total_cpu = sum(results)
    print(f"    wall clock: {dt:.1f}s   total CPU: {total_cpu:.1f}s   "
          f"parallel speedup: {total_cpu/dt:.1f}x"
          f"  ({'✓' if total_cpu/dt > 1.5 else '✗ — NOT parallel'})")


if __name__ == "__main__":
    # Test 1: CPU-bound sleep (should be perfectly parallel)
    test(_sleep_worker, "CPU burn (1s each)", 12, 12)
    # Test 2: Game emulation
    test(_game_worker, "Neural vs Example (full game)", 6, 6)
