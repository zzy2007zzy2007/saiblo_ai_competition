"""Collect ExampleAI self-play data for behavior cloning.

Usage:
    python code/bc/collect_data.py --games 500 --workers 12
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import argparse
import os
import time
import multiprocessing as mp
from datetime import datetime
from functools import partial

import numpy as np

# ─── Worker process ──────────────────────────────────────────────

_WORKER_TEMP_DIR: str = ""


def _init_worker(temp_dir: str) -> None:
    """Pool initializer: set temp dir in each worker process."""
    global _WORKER_TEMP_DIR
    _WORKER_TEMP_DIR = temp_dir


def _worker(seed: int) -> str | None:
    """Play one ExampleAI vs ExampleAI game. Return aligned board/stats/label arrays."""
    import os as _os
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    import torch as _torch
    _torch.set_num_threads(1)

    from AI.ai_example import AI as ExampleAI
    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from SDK.backend.model import OperationType, TowerType

    temp_dir = _WORKER_TEMP_DIR

    # Reverse mapping: TowerType → class_id (0-15)
    TOWER_TO_CLASS = {
        TowerType.BASIC: 0,
        TowerType.HEAVY: 1,
        TowerType.HEAVY_PLUS: 2,
        TowerType.ICE: 3,
        TowerType.BEWITCH: 4,
        TowerType.QUICK: 5,
        TowerType.QUICK_PLUS: 6,
        TowerType.DOUBLE: 7,
        TowerType.SNIPER: 8,
        TowerType.MORTAR: 9,
        TowerType.MORTAR_PLUS: 10,
        TowerType.PULSE: 11,
        TowerType.MISSILE: 12,
        TowerType.PRODUCER_FAST: 13,
        TowerType.PRODUCER_SIEGE: 14,
        TowerType.PRODUCER_MEDIC: 15,
    }

    def op_to_class(op) -> int | None:
        """Map an Operation to action class 0-22, or None if cannot map."""
        t = op.op_type
        if t == OperationType.UPGRADE_GENERATION_SPEED:
            return 21
        if t == OperationType.UPGRADE_GENERATED_ANT:
            return 22
        if t == OperationType.USE_LIGHTNING_STORM:
            return 17
        if t == OperationType.USE_EMP_BLASTER:
            return 18
        if t == OperationType.USE_DEFLECTOR:
            return 19
        if t == OperationType.USE_EMERGENCY_EVASION:
            return 20
        if t == OperationType.DOWNGRADE_TOWER:
            return 16
        if t == OperationType.BUILD_TOWER:
            return 0
        if t == OperationType.UPGRADE_TOWER:
            return TOWER_TO_CLASS.get(op.arg1)
        return None

    p0 = ExampleAI(seed=seed)
    p1 = ExampleAI(seed=seed + 10000)
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    boards: list[np.ndarray] = []
    statss: list[np.ndarray] = []
    labels: list[int] = []

    for _ in range(MAX_ROUND):
        if state.terminal:
            break

        for player, agent in ((0, p0), (1, p1)):
            obs = agent.feature_extractor.encode_observation(
                state, player, np.zeros(agent.catalog.max_actions, dtype=np.int8)
            )
            bundle = agent.choose_bundle(state, player)
            actions = list(bundle.operations)

            # If bundle has no actions, save HOLD sample (class 23)
            if not actions:
                board_np = obs["board"].astype(np.float32)
                stats_np = obs["stats"].astype(np.float32)
                boards.append(board_np)
                statss.append(stats_np)
                labels.append(23)
                continue

            # For each action in the bundle, save (board, stats, label)
            board_np = obs["board"].astype(np.float32)
            stats_np = obs["stats"].astype(np.float32)
            for op in actions:
                cls_id = op_to_class(op)
                if cls_id is not None:
                    boards.append(board_np)
                    statss.append(stats_np)
                    labels.append(cls_id)

        # Advance game
        ops0 = p0.choose_operations(state, 0)
        ops1 = p1.choose_operations(state, 1)
        state.resolve_turn(ops0, ops1)

    # If no data collected (game ended too quickly, no valid actions), return None
    if len(labels) == 0:
        print(f"  [worker {seed:4d}] no valid actions, skipping")
        return None

    n = len(labels)
    out_path = Path(temp_dir) / f"game_{seed:04d}.npz"
    np.savez_compressed(out_path,
                         board=np.stack(boards, axis=0),
                         stats=np.stack(statss, axis=0),
                         class_label=np.array(labels, dtype=np.int64))
    print(f"  [worker {seed:4d}] done — {n} samples ({state.round_index}r) → {out_path.name}")
    return str(out_path)


# ─── Main ────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Collect ExampleAI self-play data for BC")
    parser.add_argument("--games", type=int, default=500, help="number of games to play")
    parser.add_argument("--workers", type=int, default=12, help="parallel processes")
    parser.add_argument("--out", type=str, default=None, help="output .npz path")
    parser.add_argument("--temp-dir", type=str, default=None,
                        help="temp directory for per-game files (default: auto-create & cleanup)")
    args = parser.parse_args()

    # Temp directory for per-game files
    temp_dir = Path(args.temp_dir) if args.temp_dir else Path(f"_bc_tmp_{os.getpid()}")
    temp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    pool_size = min(args.workers, args.games)
    with mp.Pool(pool_size, initializer=_init_worker, initargs=(str(temp_dir),)) as pool:
        raw_results = pool.map(_worker, range(args.games))

    # Collect result files, filter out None
    result_files = [r for r in raw_results if r is not None]
    if not result_files:
        print("ERROR: no valid data collected from any game!")
        return

    # Merge: load one file at a time, accumulate
    total_n = 0
    class_counts: dict[int, int] = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else Path(f"bc_data_{ts}.npz")

    # First pass: count total samples and class distribution
    for fp in result_files:
        with np.load(fp) as data:
            n = len(data["class_label"])
            total_n += n
            for lbl in data["class_label"]:
                class_counts[int(lbl)] = class_counts.get(int(lbl), 0) + 1

    # Second pass: allocate and fill
    all_boards = np.empty((total_n, 28, 19, 19), dtype=np.float32)
    all_stats = np.empty((total_n, 42), dtype=np.float32)
    all_labels = np.empty(total_n, dtype=np.int64)
    offset = 0
    for fp in result_files:
        with np.load(fp) as data:
            n = len(data["class_label"])
            all_boards[offset:offset + n] = data["board"]
            all_stats[offset:offset + n] = data["stats"]
            all_labels[offset:offset + n] = data["class_label"]
            offset += n

    np.savez_compressed(out_path, board=all_boards, stats=all_stats, class_label=all_labels)

    # Cleanup temp files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    failed = args.games - len(result_files)
    print(f"\nCollected {total_n:,} samples from {len(result_files)} games ({failed} failed) ({time.time()-t0:.1f}s)")
    print(f"Saved: {out_path}")
    print(f"  board shape:     {all_boards.shape}")
    print(f"  stats shape:     {all_stats.shape}")
    print(f"  class_label:     {all_labels.shape}")
    dist = [class_counts.get(i, 0) for i in range(24)]
    print(f"  class_dist:      {dist}")


if __name__ == "__main__":
    main()
