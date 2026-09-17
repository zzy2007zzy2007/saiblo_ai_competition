"""Search-vs-search comparison of two checkpoints (full system test).

Both sides play with the bundle MCTS at the same config (default 128 iters /
depth 4, t_class 0.5 / t_pos 0.3 — matching self-play).  Alternates which
model is P0, reports A's win rate.  This is the definitive test of whether
AlphaZero training improved the full policy+value system.

Usage:
    python code/my_ai/az_intent/az_search_vs_search.py \
        --a .../az_az3.pt --b .../gen0120_warm.pt \
        --games 6 --workers 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _worker(args: tuple) -> dict:
    (a_path, b_path, seed, iterations, max_depth_rounds, t_class, t_pos,
     native_engine, flip, a_rel_abs, b_rel_abs, a_tanh, b_tanh) = args
    import torch
    torch.set_num_threads(1)

    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from SDK.utils.features import FeatureExtractor
    from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt, make_initial_state
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    # Each side gets its OWN value-readout config: a head trained on RELATIVE
    # labels needs raw/scale + d_t/HP_SCALE (and tanh off), while an absolute-label
    # head is used as-is.  Without this the comparison would be unfair.
    _, net_fn_a = make_net_fn_from_ckpt(a_path, feat, value_tanh=a_tanh,
                                        value_rel_to_abs=a_rel_abs)
    _, net_fn_b = make_net_fn_from_ckpt(b_path, feat, value_tanh=b_tanh,
                                        value_rel_to_abs=b_rel_abs)
    mcts_a = BundleMCTS(net_fn_a, iterations=iterations,
                        max_depth_rounds=max_depth_rounds,
                        t_class=t_class, t_pos=t_pos, seed=seed)
    mcts_b = BundleMCTS(net_fn_b, iterations=iterations,
                        max_depth_rounds=max_depth_rounds,
                        t_class=t_class, t_pos=t_pos, seed=seed + 1)

    # alternate which model is P0; --flip-sides inverts the assignment so the SAME
    # seed can be replayed with swapped sides -> pairs with the unflipped run
    a_is_p0 = (seed % 2 == 0) != bool(flip)
    mcts = [mcts_a, mcts_b] if a_is_p0 else [mcts_b, mcts_a]
    state = make_initial_state(seed, native_engine)
    for _ in range(512):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            res = mcts[player].search(state, player, temperature=0.0)
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in res.chosen_bundle
            ]
            state.apply_operation_list(player, ops)
        if player == 1 and not state.terminal:
            state.advance_round()

    # score from model_a's perspective
    a_player = 0 if a_is_p0 else 1
    if state.terminal and state.winner is not None:
        if state.winner == a_player:
            score = 1.0
        elif state.winner == 1 - a_player:
            score = 0.0
        else:
            score = 0.5
    else:
        hp_a = state.bases[a_player].hp
        hp_b = state.bases[1 - a_player].hp
        score = 0.5 if hp_a == hp_b else (1.0 if hp_a > hp_b else 0.0)
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[score]
    print(f"  seed={seed:3d} {tag} ({'A=P0' if a_is_p0 else 'A=P1'}) "
          f"hp={[b.hp for b in state.bases]} rounds={state.round_index}", flush=True)
    return {"score": score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search-vs-search comparison (full system)")
    parser.add_argument("--a", required=True, help="model A (e.g. trained)")
    parser.add_argument("--b", required=True, help="model B (e.g. pre-training)")
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--max-depth-rounds", type=int, default=4)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=1.0,
                        help="position sampling temperature (z-scored over legal cells); "
                             "see docs/az_t_pos_default_fix.md")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--native-engine", action="store_true",
                        help="use the C++ engine (native_game) for the game simulation")
    parser.add_argument("--a-rel-to-abs", type=float, default=0.0,
                        help="A: if A's value head was trained on RELATIVE labels, pass its "
                             "label_scale here (value = raw/scale + d_t/HP_SCALE)")
    parser.add_argument("--b-rel-to-abs", type=float, default=0.0, help="same, for B")
    parser.add_argument("--a-no-tanh", action="store_true", help="A: keep raw value (no tanh)")
    parser.add_argument("--b-no-tanh", action="store_true", help="B: keep raw value (no tanh)")
    parser.add_argument("--flip-sides", action="store_true",
                        help="invert the seed%%2 side assignment. Running the SAME "
                             "seeds with this flag replays every game with swapped "
                             "sides, turning the two runs into 64 matched PAIRS "
                             "(cancels the P0/P1 asymmetry within each pair)")
    args = parser.parse_args()

    jobs = [(args.a, args.b, args.seed + s, args.iterations, args.max_depth_rounds,
             args.t_class, args.t_pos, args.native_engine, args.flip_sides,
             args.a_rel_to_abs, args.b_rel_to_abs,
             not args.a_no_tanh, not args.b_no_tanh)
            for s in range(args.games)]
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
    print(f"\n=== {a_name} (A) vs {b_name} (B) [search {args.iterations}/{args.max_depth_rounds}]: "
          f"A win rate = {win/len(scores):.1%} ({win}W/{draw}D/{loss}L) ===")


if __name__ == "__main__":
    main()
