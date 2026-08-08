"""Benchmark the C++ engine's Python-callable speed vs the Python engine.

Measures:
  - per apply_operation_list call (pybind11 overhead + op application)
  - per advance_round call (round resolution)
  - a full game to termination
  - the same via the Python engine for comparison

Usage:
    python code/cpp_engine/benchmark.py
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO / "Ant-Game", _REPO / "code", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from SDK.backend.engine import GameState
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType

import native_game
from test_consistency import random_op, MAX_OPS

SEED = 2024


def timeit(fn, n):
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main() -> None:
    rng = random.Random(SEED * 1000 + 7)
    rounds_ops = [
        [[random_op(rng) for _ in range(rng.randint(0, MAX_OPS))],
         [random_op(rng) for _ in range(rng.randint(0, MAX_OPS))]]
        for _ in range(200)
    ]

    # ── C++ engine ──
    g = native_game.NativeGame()
    g.init_game(seed=SEED, movement_policy="enhanced", cold=True)

    # per apply_operation_list (ops already generated; reuse round-0 ops)
    ops0 = rounds_ops[0][0]
    t_apply = timeit(lambda: g.apply_operation_list(0, ops0), 2000)
    # per advance_round (empty-ish round)
    t_adv = timeit(lambda: g.advance_round(), 2000)

    # full game
    g2 = native_game.NativeGame()
    g2.init_game(seed=SEED, movement_policy="enhanced", cold=True)
    t0 = time.perf_counter()
    for ro in rounds_ops:
        for p, ops in enumerate(ro):
            g2.apply_operation_list(p, ops)
        if not g2.advance_round():
            break
    game_s = time.perf_counter() - t0
    rounds_played = g2.round()

    # ── Python engine (same ops) ──
    state = GameState.initial(seed=SEED, cold_handle_rule_illegal=True)
    t_apply_py = timeit(
        lambda: state.apply_operation_list(0, [Operation(OperationType(t), a, b) for t, a, b in ops0]),
        500,
    )
    t_adv_py = timeit(lambda: state.advance_round(), 500)
    t0 = time.perf_counter()
    for ro in rounds_ops:
        for p, ops in enumerate(ro):
            state.apply_operation_list(p, [Operation(OperationType(t), a, b) for t, a, b in ops])
        state.advance_round()
        if state.terminal:
            break
    game_py_s = time.perf_counter() - t0

    print(f"apply_operation_list:  C++ {t_apply*1e6:8.1f} us   Python {t_apply_py*1e6:8.1f} us   "
          f"{t_apply_py/t_apply:5.1f}x")
    print(f"advance_round:         C++ {t_adv*1e6:8.1f} us   Python {t_adv_py*1e6:8.1f} us   "
          f"{t_adv_py/t_adv:5.1f}x")
    print(f"full game ({rounds_played} rounds): C++ {game_s:6.2f} s   "
          f"Python {game_py_s:6.2f} s   {game_py_s/game_s:5.1f}x")


if __name__ == "__main__":
    main()
