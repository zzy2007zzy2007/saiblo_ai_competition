"""Evaluate the official AlphaZero scaffold's PolicyValueNet + PriorGuidedMCTS vs ExampleAI.

Usage:
    python code/test_match/eval_official_az.py --checkpoint training_history/az_full/az_full.npz --games 8 --workers 4
    python code/test_match/eval_official_az.py --random --games 4   # random-init baseline

The scaffold's PolicyValueNet is pure numpy (no torch), so workers are cheap.
Scoring: uses the engine's terminal winner when set (includes the 512-round
tie-break), otherwise falls back to base-HP comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np


def _build_agent(ckpt_path: str | None, iterations: int, max_depth: int, seed: int):
    from SDK.alphazero import PolicyValueNet, PriorGuidedMCTS, SearchConfig, build_policy_value_net
    from SDK.utils.features import FeatureExtractor
    from SDK.utils.actions import ActionCatalog

    if ckpt_path is None:
        feat = FeatureExtractor()
        network = build_policy_value_net(feat, action_dim=feat.max_actions)
        print(f"[eval] random-init network (obs_dim={network.obs_dim})", flush=True)
    else:
        network = PolicyValueNet.from_checkpoint(ckpt_path)
        print(f"[eval] loaded checkpoint {ckpt_path} (obs_dim={network.obs_dim})", flush=True)
    config = SearchConfig(iterations=iterations, max_depth=max_depth, seed=seed)
    search = PriorGuidedMCTS(model=network, search_config=config)
    return search


def _play_one(search, seed: int, max_rounds: int = 512) -> dict:
    from SDK.backend.engine import GameState
    from SDK.utils.turns import DecisionContext
    from AI.ai_example import AI as ExampleAI

    opp = ExampleAI(seed=seed)
    our_player = seed % 2
    opp_player = 1 - our_player
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)

    for _ in range(max_rounds):
        if state.terminal:
            break
        ctx = DecisionContext.for_player(our_player)
        result = search.search(state, our_player, context=ctx, temperature=0.0, add_root_noise=False)
        ops_us = list(result.bundle.operations)
        ops_opp = opp.choose_operations(state, opp_player)
        if our_player == 0:
            state.resolve_turn(ops_us, ops_opp)
        else:
            state.resolve_turn(ops_opp, ops_us)

    # Score: terminal winner (engine decides, incl. 512-cap tie-break) else HP compare
    if state.terminal and state.winner is not None:
        if state.winner == our_player:
            score = 1.0
        elif state.winner == opp_player:
            score = 0.0
        else:
            score = 0.5
    else:
        hp_us = state.bases[our_player].hp
        hp_opp = state.bases[opp_player].hp
        score = 0.5 if hp_us == hp_opp else (1.0 if hp_us > hp_opp else 0.0)

    hp_us = state.bases[our_player].hp
    hp_opp = state.bases[opp_player].hp
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[score]
    side = "1st" if our_player == 0 else "2nd"
    print(f"  [{side}] seed={seed:3d}  {tag}  us={int(hp_us):2d}  opp={int(hp_opp):2d}  winner={state.winner}", flush=True)
    return {"score": score, "rounds": state.round_index}


def _worker(args: tuple) -> dict:
    ckpt_path, iterations, max_depth, seed = args
    search = _build_agent(ckpt_path, iterations, max_depth, seed)
    return _play_one(search, seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval official scaffold AlphaZero agent vs ExampleAI")
    parser.add_argument("--checkpoint", type=str, default=None, help=".npz checkpoint from SDK/train_mcts.py")
    parser.add_argument("--random", action="store_true", help="use a fresh random-init network (baseline)")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.random:
        args.checkpoint = None
    if args.checkpoint is None and not args.random:
        parser.error("must provide --checkpoint or --random")

    worker_args = [(args.checkpoint, args.iterations, args.max_depth, args.seed + s) for s in range(args.games)]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, worker_args)
    else:
        results = [_worker(a) for a in worker_args]

    scores = [r["score"] for r in results]
    win = sum(1 for s in scores if s == 1.0)
    draw = sum(1 for s in scores if s == 0.5)
    loss = sum(1 for s in scores if s == 0.0)
    avg_rounds = np.mean([r["rounds"] for r in results])
    label = "random-init" if args.checkpoint is None else Path(args.checkpoint).name
    print(f"\n=== {label} vs ExampleAI: {win}W / {draw}D / {loss}L  (win_rate={win / len(scores):.2%}, avg_rounds={avg_rounds:.0f}) ===")


if __name__ == "__main__":
    main()
