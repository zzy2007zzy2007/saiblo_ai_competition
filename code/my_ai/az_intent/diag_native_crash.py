"""Diagnose native_game (C++ engine) crashes during MCTS play.

Plays games (az_r14 search vs rule_v4) on the C++ engine, logging every
native_game call to a file BEFORE it happens (flush per line).  If the C++
engine segfaults, the last line identifies the exact crashing call.

Usage:
    python code/my_ai/az_intent/diag_native_crash.py [seed_start] [n_games]
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np

LOG = Path("training_history/az_intent/native_crash_diag.log")


def log(line: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def play_one(seed: int, iterations: int, max_depth_rounds: int) -> None:
    import torch
    torch.set_num_threads(1)
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, make_initial_state
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.az_intent.game_state_facade import GameStateFacade

    feat = FeatureExtractor(max_actions=96)
    _, net_fn = make_net_fn_from_ckpt("training_history/az_intent/az_r14.pt", feat)
    mcts = BundleMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds,
                      k=24, sample_mult=15, t_class=0.5, t_pos=0.3, seed=seed)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "其他版本ai" / "rule_v4"))
    from ai import AI as OppAI
    opp = OppAI(seed=seed)
    our_player = seed % 2

    state = make_initial_state(seed, native_engine=True)
    log(f"=== seed {seed} our_player={our_player} ===")
    for round_idx in range(512):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            if player == our_player:
                log(f"r{round_idx} search p{player}")
                res = mcts.search(state, player, temperature=0.0)
                ops = [Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                       for k in res.chosen_bundle]
            else:
                log(f"r{round_idx} rule_v4 p{player}")
                ops = opp.choose_operations(state, player)
            log(f"r{round_idx} apply p{player} ops={[(int(o.op_type), o.arg0, o.arg1) for o in ops]}")
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            log(f"r{round_idx} advance_round")
            state.advance_round()
    log(f"=== seed {seed} END winner={state.winner} hp={[b.hp for b in state.bases]} ===")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=607)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--max-depth-rounds", type=int, default=2)
    args = parser.parse_args()

    if LOG.exists():
        LOG.unlink()
    log(f"diag start: seed={args.seed} games={args.games} iters={args.iterations}")
    for g in range(args.games):
        seed = args.seed + g
        log(f"--- game {g} seed {seed} begin ---")
        play_one(seed, args.iterations, args.max_depth_rounds)
        log(f"--- game {g} seed {seed} completed ---")


if __name__ == "__main__":
    main()
