"""DIAGNOSTIC: C++ engine (native_game) vs Python engine, turn-by-turn.

NOTE (2026-08-08): the C++ engine is the AUTHORITATIVE judge.  The Python
SDK drifts from it (99/361 map-validity cells differ, plus legality checks
like build-highland / weapon-VOID).  This script documents WHERE the two
engines diverge — it is NOT the M1 acceptance gate (the gate is: the copy is
byte-identical to the official game/ sources, verified by `diff`).

Usage:
    python code/cpp_engine/test_consistency.py [--rounds 120]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for p in (_REPO / "Ant-Game", _REPO / "code", Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from SDK.backend.engine import GameState
from SDK.backend.model import Operation
from SDK.utils.constants import OperationType
import native_game

OP_TYPES = [11, 12, 13, 21, 22, 23, 24, 31, 32]
VALID_TOWER_TYPES = [1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43]
MAX_OPS = 4


def random_op(rng):
    t = rng.choice(OP_TYPES)
    if t == 12:  # upgrade: arg1 must be a valid TowerType
        return [t, rng.randint(0, 30), rng.choice(VALID_TOWER_TYPES)]
    return [t, rng.randint(0, 18), rng.randint(0, 18)]


def python_snapshot(state):
    towers = sorted((t.tower_id, t.x, t.y, t.player, int(t.tower_type), t.hp,
                     t.stats.max_hp) for t in state.towers)
    ants = sorted((a.ant_id, a.x, a.y, a.player, a.hp, int(a.kind))
                  for a in state.ants)
    return (state.bases[0].hp, state.bases[1].hp, towers, ants)


def cpp_snapshot(g):
    towers = sorted(tuple(t) for t in g.tower_snapshot())
    ants = sorted(tuple(a) for a in g.ant_snapshot())
    return (g.base_hp(0), g.base_hp(1), towers, ants)


def run_one(seed: int, rounds: int, rng: random.Random):
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    g = native_game.NativeGame()
    g.init_game(seed=seed, movement_policy="enhanced", cold=True)
    for r in range(rounds):
        for p in (0, 1):
            raw = [random_op(rng) for _ in range(rng.randint(0, MAX_OPS))]
            py_ops = [Operation(OperationType(t), a, b) for t, a, b in raw]
            state.apply_operation_list(p, py_ops)
            g.apply_operation_list(p, raw)
        state.advance_round()
        g.advance_round()
        ps, cs = python_snapshot(state), cpp_snapshot(g)
        if ps != cs:
            return r, ps, cs
    return None, python_snapshot(state), cpp_snapshot(g)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--seeds", type=str, default="0,1,7,42,123,2024")
    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    ok = 0
    for seed in seeds:
        rng = random.Random(seed * 1000 + 7)
        bad, ps, cs = run_one(seed, args.rounds, rng)
        if bad is None:
            print(f"seed {seed:6d}: {args.rounds} rounds OK")
            ok += 1
        else:
            print(f"seed {seed:6d}: MISMATCH at round {bad}")
            print(f"  python: {ps}")
            print(f"  cpp  : {cs}")
    print(f"\n{ok}/{len(seeds)} seeds fully consistent "
          f"({args.rounds} rounds each)")
    sys.exit(0 if ok == len(seeds) else 1)


if __name__ == "__main__":
    main()
