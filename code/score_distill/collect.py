"""Collect game data with ActionCatalog score maps for score regression training.

Usage:
    python code/score_distill/collect.py --games 500 --workers 12 --out-dir score_data

Output:
    score_data/gen_0000_*.npz  — each file contains board, stats, action_map,
                                 score_map (24,19,19), class_scores (24,)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
for p in (_REPO, Path(__file__).resolve().parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import os, time, argparse, multiprocessing as mp
import numpy as np
import torch


def _worker(seed: int, out_dir: str) -> dict:
    """Play one game vs ExampleAI, collect score_map at every step."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from SDK.utils.features import FeatureExtractor
    from AI.ai_example import AI as ExampleAI
    from my_ai.agent import NeuralAgent
    from my_ai.network import create_model
    from score_distill.score_ops import build_score_map

    # Use a dummy model to get action_map from ExampleAI's choices
    # (or we can just skip model output and only save score_map + board)
    our_player = seed % 2
    opp_player = 1 - our_player

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    ai = ExampleAI(seed=seed)
    feat = FeatureExtractor()

    boards, statss = [], []
    score_maps, class_scores_list = [], []

    for turn in range(MAX_ROUND):
        if state.terminal:
            break

        # Record board state
        feat_out = feat.encode_observation(state, our_player, np.zeros(64))
        boards.append(feat_out["board"].copy())
        statss.append(feat_out["stats"].copy())

        # Compute ActionCatalog score map
        score_map, class_scores = build_score_map(state, our_player)
        score_maps.append(score_map)
        class_scores_list.append(class_scores)

        # Advance game (both players)
        our_ops = ai.choose_operations(state, our_player)
        opp_ops = ai.choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(our_ops, opp_ops)
        else:
            state.resolve_turn(opp_ops, our_ops)

    if not boards:
        return {"seeds": [seed], "frames": 0}

    # Save to npz
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed{seed:05d}.npz"

    # Also build a simple action_map (dummy uniform) for compatibility
    T = len(boards)
    dummy_action_map = np.ones((T, 24, 19, 19), dtype=np.float16) / (19*19)
    score_map_arr = np.stack(score_maps, axis=0).astype(np.float16)       # (T, 24, 19, 19)
    class_scores_arr = np.stack(class_scores_list, axis=0).astype(np.float32)  # (T, 24)

    np.savez_compressed(path,
        board=np.stack(boards, axis=0).astype(np.float16),
        stats=np.stack(statss, axis=0).astype(np.float16),
        action_map=dummy_action_map,
        score_map=score_map_arr,
        class_scores=class_scores_arr,
    )
    return {"seeds": [seed], "frames": T}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=500, help="number of games")
    parser.add_argument("--workers", type=int, default=8, help="parallel workers")
    parser.add_argument("--out-dir", type=str, default="score_data",
                        help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with mp.Pool(args.workers) as pool:
        results = pool.starmap(_worker, [
            (s, str(out_dir)) for s in range(args.games)
        ])

    total_frames = sum(r["frames"] for r in results)
    elapsed = time.time() - t0
    print(f"\nCollected {args.games} games, {total_frames} frames "
          f"in {elapsed:.1f}s ({total_frames/elapsed:.0f} frames/s)")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
