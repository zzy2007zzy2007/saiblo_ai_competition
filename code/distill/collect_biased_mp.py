"""Collect BC data using BiasedActionCatalog - multiprocessing with controlled threads.

Key settings:
  - OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 to prevent thread explosion
  - workers=4 (balance of speed vs memory)
  - Saves each game as individual .npz file
"""
from __future__ import annotations
import sys, time, argparse, multiprocessing as mp
from pathlib import Path

# Must set thread env vars BEFORE any numpy/torch import
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from SDK.backend.engine import GameState
from SDK.utils.constants import MAX_ROUND
from SDK.utils.features import FeatureExtractor
from AI.ai_example import AI as ExampleAI

from my_ai.decoder import (
    CHANNEL_TO_TOWER_TYPE, upgrade_step, NUM_CLASSES,
)
from my_ai.actions_biased import BiasedActionCatalog

OP_HOLD = 23


def _compute_class_scores(bundles, state, player) -> np.ndarray:
    from SDK.utils.constants import OperationType
    score_lookup: dict[tuple[int, int, int], float] = {}
    for b in bundles:
        if len(b.operations) == 1:
            op = b.operations[0]
            key = (int(op.op_type), op.arg0, op.arg1)
            if key not in score_lookup or b.score > score_lookup[key]:
                score_lookup[key] = b.score
    scores = np.full(24, -1.0, dtype=np.float32)
    def _update(cls_id: int, score_value: float):
        if score_value > scores[cls_id]:
            scores[cls_id] = score_value
    for (op_type, arg0, arg1), score_value in score_lookup.items():
        if op_type == int(OperationType.BUILD_TOWER):
            for cls in range(16):
                _update(cls, score_value)
        elif op_type == int(OperationType.UPGRADE_TOWER):
            tower_id, step_val = arg0, arg1
            for t in state.towers_of(player):
                if t.tower_id == tower_id:
                    for cls in range(1, 16):
                        tt = CHANNEL_TO_TOWER_TYPE[cls]
                        s = upgrade_step(t.tower_type, tt)
                        if s is not None and int(s) == step_val:
                            _update(cls, score_value)
                    break
        elif op_type == int(OperationType.DOWNGRADE_TOWER):
            _update(16, score_value)
        elif op_type == int(OperationType.USE_LIGHTNING_STORM):
            _update(17, score_value)
        elif op_type == int(OperationType.USE_EMP_BLASTER):
            _update(18, score_value)
        elif op_type == int(OperationType.USE_DEFLECTOR):
            _update(19, score_value)
        elif op_type == int(OperationType.USE_EMERGENCY_EVASION):
            _update(20, score_value)
        elif op_type == int(OperationType.UPGRADE_GENERATION_SPEED):
            _update(21, score_value)
        elif op_type == int(OperationType.UPGRADE_GENERATED_ANT):
            _update(22, score_value)
    return scores


def _op_to_label(op, state, player) -> tuple[int, int, int]:
    from SDK.utils.constants import OperationType
    ot = op.op_type
    if ot == OperationType.BUILD_TOWER:
        return (0, op.arg0, op.arg1)
    if ot == OperationType.UPGRADE_TOWER:
        tower_id = op.arg0
        target = op.arg1
        for t in state.towers_of(player):
            if t.tower_id == tower_id:
                for ch in range(1, 16):
                    tt = CHANNEL_TO_TOWER_TYPE[ch]
                    step = upgrade_step(t.tower_type, tt)
                    if step is not None and int(step) == target:
                        return (ch, t.x, t.y)
        return (OP_HOLD, -1, -1)
    if ot == OperationType.DOWNGRADE_TOWER:
        for t in state.towers_of(player):
            if t.tower_id == op.arg0:
                return (16, t.x, t.y)
        return (OP_HOLD, -1, -1)
    if ot == OperationType.USE_LIGHTNING_STORM:
        return (17, op.arg0, op.arg1)
    if ot == OperationType.USE_EMP_BLASTER:
        return (18, op.arg0, op.arg1)
    if ot == OperationType.USE_DEFLECTOR:
        return (19, op.arg0, op.arg1)
    if ot == OperationType.USE_EMERGENCY_EVASION:
        return (20, op.arg0, op.arg1)
    if ot == OperationType.UPGRADE_GENERATION_SPEED:
        return (21, -1, -1)
    if ot == OperationType.UPGRADE_GENERATED_ANT:
        return (22, -1, -1)
    return (OP_HOLD, -1, -1)


def collect_game(seed: int, max_actions: int = 9999) -> list[dict]:
    catalog = BiasedActionCatalog(max_actions=max_actions)
    extractor = FeatureExtractor(max_actions=max_actions)
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    our_player = seed % 2
    opp_player = 1 - our_player
    opp = ExampleAI(seed=seed + 9999)
    turns_data: list[dict] = []
    for turn in range(MAX_ROUND):
        if state.terminal:
            break
        bundles = catalog.build(state, our_player, rerank=False)
        best = max(bundles, key=lambda b: (b.score, -len(b.operations)))
        class_scores = _compute_class_scores(bundles, state, our_player)
        if len(best.operations) > 0:
            cls, x, y = _op_to_label(best.operations[0], state, our_player)
        else:
            cls, x, y = (OP_HOLD, -1, -1)
        obs = extractor.encode_observation(state, our_player,
                                           np.zeros(max_actions, dtype=np.float32))
        board = obs["board"].astype(np.float32)
        stats = obs["stats"].astype(np.float32)
        action_map = np.zeros((NUM_CLASSES, 19, 19), dtype=np.float32)
        if cls != OP_HOLD and x >= 0 and y >= 0:
            action_map[cls, x, y] = 10.0
        head_logits = np.full((3, 24), -10.0, dtype=np.float32)
        head_logits[:, cls] = 10.0
        class_label = np.full(3, cls, dtype=np.int64)
        turns_data.append({
            "board": board, "stats": stats, "class_": class_label,
            "action_map": action_map, "head_logits": head_logits,
            "class_scores": class_scores,
        })
        our_ops = best.operations
        opp_ops = opp.choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(our_ops, opp_ops)
        else:
            state.resolve_turn(opp_ops, our_ops)
    return turns_data


def _worker(seed: int, out_dir: str) -> dict:
    # Worker process: ensure no thread explosion
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    out_path = Path(out_dir)
    turns = collect_game(seed)
    if not turns:
        return {"seed": seed, "turns": 0, "file": ""}
    board = np.stack([t["board"] for t in turns], axis=0)
    stats = np.stack([t["stats"] for t in turns], axis=0)
    class_ = np.stack([t["class_"] for t in turns], axis=0)
    action_map = np.stack([t["action_map"] for t in turns], axis=0)
    head_logits = np.stack([t["head_logits"] for t in turns], axis=0)
    class_scores = np.stack([t["class_scores"] for t in turns], axis=0)
    F16_MAX = 65504.0
    path = out_path / f"distill_seed{seed}.npz"
    np.savez_compressed(
        path,
        board=board.astype(np.float16),
        stats=stats.astype(np.float16),
        class_=class_,
        action_map=np.clip(action_map, -F16_MAX, F16_MAX).astype(np.float16),
        head_logits=np.clip(head_logits, -F16_MAX, F16_MAX).astype(np.float16),
        class_scores=class_scores.astype(np.float16),
    )
    return {"seed": seed, "turns": len(turns), "file": str(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="distill_data_biased")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Set thread limits in main process too
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting {args.games} games, {args.workers} workers → {out_dir}")
    print(f"  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}")
    t0 = time.time()

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = pool.starmap(
                _worker,
                [(args.seed + g, str(out_dir)) for g in range(args.games)])
    else:
        results = [_worker(args.seed + g, str(out_dir)) for g in range(args.games)]

    total_turns = sum(r["turns"] for r in results)
    elapsed = time.time() - t0
    saved = sum(1 for r in results if r["file"])
    print(f"\nDone: {saved}/{args.games} games, {total_turns} turns, {elapsed:.1f}s")


if __name__ == "__main__":
    main()
