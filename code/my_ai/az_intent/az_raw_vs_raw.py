"""Quick raw-vs-raw comparison of two checkpoints (no search).

Plays self-match games where both sides use their raw (argmax decode) policy,
alternating which model is P0, and reports win rates.  Used to test whether
AlphaZero training improved the policy (trained vs pre-training).

Usage:
    python code/my_ai/az_intent/az_raw_vs_raw.py --a .../trained.pt --b .../gen0120_warm.pt \
        --games 10 --workers 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch


def _worker(args: tuple) -> dict:
    a_path, b_path, seed = args
    import torch
    torch.set_num_threads(1)
    from SDK.backend.engine import GameState
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import load_model_from_ckpt
    from my_ai.decoder import decode_network_output

    model_a = load_model_from_ckpt(a_path)
    model_a.eval()
    model_b = load_model_from_ckpt(b_path)
    model_b.eval()
    feat = FeatureExtractor(max_actions=96)

    # alternate which model is P0
    models = [model_a, model_b] if seed % 2 == 0 else [model_b, model_a]
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for _ in range(512):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = models[player](torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                     torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            ops = decode_network_output(out, state, player, temperature=0.0, intent_decoding=True)
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()

    # score from model_a's perspective
    if state.terminal and state.winner is not None:
        a_player = 0 if seed % 2 == 0 else 1
        if state.winner == a_player:
            score = 1.0
        elif state.winner == 1 - a_player:
            score = 0.0
        else:
            score = 0.5
    else:
        a_player = 0 if seed % 2 == 0 else 1
        hp_a = state.bases[a_player].hp
        hp_b = state.bases[1 - a_player].hp
        score = 0.5 if hp_a == hp_b else (1.0 if hp_a > hp_b else 0.0)
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[score]
    print(f"  seed={seed:3d} {tag} ({'A=P0' if seed % 2 == 0 else 'A=P1'}) "
          f"hp={[b.hp for b in state.bases]} rounds={state.round_index}", flush=True)
    return {"score": score}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="model A (e.g. trained)")
    parser.add_argument("--b", required=True, help="model B (e.g. pre-training)")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    jobs = [(args.a, args.b, args.seed + s) for s in range(args.games)]
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    scores = [r["score"] for r in results]
    win = sum(1 for s in scores if s == 1.0)
    draw = sum(1 for s in scores if s == 0.5)
    loss = sum(1 for s in scores if s == 0.0)
    a_name = Path(args.a).name
    b_name = Path(args.b).name
    print(f"\n=== {a_name} (A) vs {b_name} (B): A win rate = {win/len(scores):.1%} "
          f"({win}W/{draw}D/{loss}L) ===")


if __name__ == "__main__":
    main()
