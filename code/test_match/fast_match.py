#!/usr/bin/env python3
"""
Fast match runner for RL training.
- Parallel multi-processing support
- Skips unnecessary serialization when not needed
- Direct state access for neural network input
"""

import sys
import time
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path("d:/2026智能体大赛_新2/Ant-Game").resolve()
sys.path.insert(0, str(REPO_ROOT))

from SDK.backend.engine import GameState
from SDK.backend.model import Operation
from SDK.utils.constants import MAX_ROUND, OperationType


# ── lightweight policy interface ──────────────────────────────────────

class Policy:
    """Base class for fast policies. Subclass and implement __call__()."""
    def __call__(self, state: GameState, player: int) -> list[Operation]:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Truly random policy that picks random valid operations."""
    def __init__(self, seed: int = 0):
        import random
        self.rng = random.Random(seed)

    def __call__(self, state: GameState, player: int) -> list[Operation]:
        ops = []
        for _ in range(self.rng.randint(1, 4)):
            x = self.rng.randint(0, 18)
            y = self.rng.randint(0, 18)
            op = Operation(OperationType.BUILD_TOWER, x, y)
            if state.can_apply_operation(player, op, ops):
                ops.append(op)
        return ops


class NoOpPolicy(Policy):
    """No-op policy for baseline speed testing."""
    def __call__(self, state: GameState, player: int) -> list[Operation]:
        return []


# ── fast match runner ──────────────────────────────────────────────────

@dataclass
class MatchResult:
    rounds: int
    winner: int | None
    hp: list[int]
    coins: list[int]
    tower_count: list[int]
    ant_count: int
    die_count: list[int]
    duration: float


def run_fast_match(
    policy0: Policy,
    policy1: Policy,
    seed: int = 7,
    max_rounds: int = MAX_ROUND,
    skip_public_state: bool = True,
) -> MatchResult:
    """Run a single match as fast as possible."""
    state = GameState.initial(seed=seed, movement_policy="enhanced")
    t0 = time.perf_counter()

    for round_idx in range(max_rounds):
        if state.terminal:
            break

        ops0 = policy0(state, 0)
        ops1 = policy1(state, 1)
        resolution = state.resolve_turn(ops0, ops1)

        if not skip_public_state:
            _ = state.to_public_round_state()

        if resolution.terminal:
            break

    dt = time.perf_counter() - t0
    return MatchResult(
        rounds=round_idx + 1,
        winner=state.winner,
        hp=[state.bases[0].hp, state.bases[1].hp],
        coins=list(state.coins),
        tower_count=[
            len([t for t in state.towers if t.player == 0]),
            len([t for t in state.towers if t.player == 1]),
        ],
        ant_count=len(state.ants),
        die_count=list(state.die_count),
        duration=dt,
    )


def _run_match_wrapper(args: tuple) -> MatchResult:
    """Wrapper for multiprocessing: unpacks args and calls run_fast_match."""
    policy0_cls, policy1_cls, seed, skip_pub = args
    return run_fast_match(policy0_cls(), policy1_cls(), seed=seed, skip_public_state=skip_pub)


def run_parallel_matches(
    policy0_cls: type,
    policy1_cls: type,
    num_matches: int,
    start_seed: int = 0,
    skip_public_state: bool = True,
    num_workers: int | None = None,
) -> list[MatchResult]:
    """Run multiple matches in parallel using multiprocessing."""
    if num_workers is None:
        num_workers = mp.cpu_count()

    args_list = [
        (policy0_cls, policy1_cls, start_seed + i, skip_public_state)
        for i in range(num_matches)
    ]

    with mp.Pool(num_workers) as pool:
        results = pool.map(_run_match_wrapper, args_list)

    return results


# ── benchmarking ────────────────────────────────────────────────────────

def benchmark():
    """Run benchmarks comparing different configurations."""
    seed = 7
    num_parallel = 8

    print("=" * 60)
    print("性能基准测试 (纯引擎速度)")
    print("=" * 60)

    # 1. Baseline: pure engine, no ops
    state = GameState.initial(seed=seed)
    t0 = time.perf_counter()
    for i in range(MAX_ROUND):
        if state.terminal:
            break
        state.resolve_turn([], [])
    dt_engine = time.perf_counter() - t0
    print(f"\n[1] 纯引擎(无操作):")
    print(f"    {i+1} 回合, {dt_engine:.3f}s, {dt_engine/(i+1)*1000:.2f}ms/回合")

    # 2. Fast match with NoOpPolicy (skip public state)
    t0 = time.perf_counter()
    r = run_fast_match(NoOpPolicy(), NoOpPolicy(), seed=seed, skip_public_state=True)
    dt = time.perf_counter() - t0
    print(f"\n[2] NoOpPolicy (skip_public=True):")
    print(f"    {r.rounds} 回合, {dt:.3f}s, {dt/r.rounds*1000:.2f}ms/回合")

    # 3. Fast match with NoOpPolicy (include public state)
    t0 = time.perf_counter()
    r = run_fast_match(NoOpPolicy(), NoOpPolicy(), seed=seed, skip_public_state=False)
    dt = time.perf_counter() - t0
    print(f"\n[3] NoOpPolicy (skip_public=False):")
    print(f"    {r.rounds} 回合, {dt:.3f}s, {dt/r.rounds*1000:.2f}ms/回合")

    # 4. Serial matches (NoOp)
    print(f"\n[4] Serial {num_parallel}x NoOpPolicy matches:")
    t0 = time.perf_counter()
    for i in range(num_parallel):
        run_fast_match(NoOpPolicy(), NoOpPolicy(), seed=seed + i, skip_public_state=True)
    dt_serial = time.perf_counter() - t0
    print(f"    {num_parallel} 场, {dt_serial:.3f}s, {dt_serial/num_parallel:.3f}s/场")

    # 5. Parallel matches (NoOp)
    print(f"\n[5] Parallel {num_parallel}x NoOpPolicy (workers={mp.cpu_count()}):")
    t0 = time.perf_counter()
    results = run_parallel_matches(
        NoOpPolicy, NoOpPolicy, num_matches=num_parallel,
        start_seed=seed, skip_public_state=True,
    )
    dt_para = time.perf_counter() - t0
    avg_dt = sum(r.duration for r in results) / len(results)
    print(f"    {num_parallel} 场, {dt_para:.3f}s, {dt_para/num_parallel:.3f}s/场")
    print(f"    加速比: {dt_serial/dt_para:.1f}x")
    print(f"    平均每场引擎耗时: {avg_dt:.3f}s")

    # 6. Estimated throughput
    matches_per_second = num_parallel / dt_para
    print(f"\n📊 预估吞吐量:")
    print(f"    并行 {num_parallel} 场: {matches_per_second:.1f} 场/秒")
    print(f"    每小时: {matches_per_second * 3600:.0f} 场")


if __name__ == "__main__":
    benchmark()
