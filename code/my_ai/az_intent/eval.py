"""P4: Evaluate an intent-space AlphaZero model vs a reference opponent.

Opponents:
  - rule_v4 (default): the strong reference — the hot-start model wins <20% vs
    it, so it is the meaningful benchmark for measuring improvement.
  - example: ExampleAI — NOT useful for our models (the hot-start already wins
    100% without search); kept for quick smoke tests.

Modes:
  - --checkpoint <trained.pt>: our AZTrainer checkpoint (model_state format)
  - --baseline-hotstart <es.pt>: hot-start source (ES format); measures the
    pre-training policy + search as the baseline the trained model must beat

Usage:
    python code/my_ai/az_intent/eval.py --checkpoint training_history/az_intent/az_intent.pt \
        --games 8 --workers 4 --opponent rule_v4
    python code/my_ai/az_intent/eval.py --baseline-hotstart .../gen_0120.pt --games 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
_RV4 = Path(__file__).resolve().parents[3] / "其他版本ai" / "rule_v4"
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np


def _worker(args: tuple) -> dict:
    ckpt_path, iterations, max_depth_rounds, seed, our_player, opponent, hotstart, no_search, select_by_prior, intent_decoding, one_head, bundle_mcts, k, c_puct, t_class, t_pos, self_raw_opponent, max_rounds, native_engine = args
    import torch
    torch.set_num_threads(1)

    from my_ai.az_intent.az_selfplay import make_initial_state
    from SDK.utils.features import FeatureExtractor

    if hotstart:
        from my_ai.az_intent.train import build_model
        model = build_model(hotstart)
    else:
        from my_ai.network import create_model
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = create_model(
            num_resblocks=ckpt.get("num_resblocks", 6),
            num_heads=ckpt.get("num_heads", 3),
            latent_dim=ckpt.get("latent_dim", 64),
            no_bn=ckpt.get("no_bn", True),
        )
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat = FeatureExtractor(max_actions=96)
    state = make_initial_state(seed, native_engine)
    opp_player = 1 - our_player

    def opp_play(player):
        if self_raw_opponent:
            # opponent = raw (no search) decode of the SAME model
            from my_ai.decoder import decode_network_output
            obs = feat.encode_observation(state, player, np.zeros(96))
            board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
            stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
            with torch.no_grad():
                out = model(board, stats)
            ops = decode_network_output(out, state, player, temperature=0.0, intent_decoding=True)
            state.apply_operation_list(player, ops)
        else:
            ops = opp.choose_operations(state, player)
            state.apply_operation_list(player, ops)

    if self_raw_opponent:
        opp = None
    elif opponent == "rule_v4":
        if str(_RV4) not in sys.path:
            sys.path.insert(0, str(_RV4))
        from ai import AI as OppAI
        opp = OppAI(seed=seed)
    else:
        from AI.ai_example import AI as OppAI
        opp = OppAI(seed=seed)

    if no_search:
        from my_ai.decoder import decode_network_output

        def our_play(player):
            obs = feat.encode_observation(state, player, np.zeros(96))
            board = torch.from_numpy(obs["board"]).unsqueeze(0).float()
            stats = torch.from_numpy(obs["stats"]).unsqueeze(0).float()
            with torch.no_grad():
                out = model(board, stats)
            net_out = out
            if one_head:
                net_out = {"action_map": out["action_map"],
                           "head1_logits": out["head1_logits"],
                           "value": out["value"]}
            ops = decode_network_output(net_out, state, player, temperature=0.0,
                                        intent_decoding=intent_decoding)
            state.apply_operation_list(player, ops)
    elif bundle_mcts:
        from SDK.backend.model import Operation
        from SDK.utils.constants import OperationType
        from my_ai.az_intent.bundle_mcts import BundleMCTS
        from my_ai.az_intent.az_selfplay import make_net_fn_from_ckpt

        _, net_fn = make_net_fn_from_ckpt(ckpt_path, feat)
        bmcts = BundleMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds,
                           k=k, c_puct=c_puct, t_class=t_class, t_pos=t_pos, seed=seed)

        def our_play(player):
            res = bmcts.search(state, player, temperature=0.0)
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in res.chosen_bundle
            ]
            state.apply_operation_list(player, ops)
    else:
        from my_ai.az_intent.mcts import IntentMCTS, intent_to_operation
        from my_ai.az_intent.train import make_net_fn

        net_fn = make_net_fn(model, feat)
        mcts = IntentMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds, seed=seed)

        def our_play(player):
            for head in (0, 1, 2):
                if state.terminal:
                    break
                res = mcts.search(state, player, head_idx=head, temperature=0.0,
                                  add_root_noise=False, select_by_prior=select_by_prior)
                op = intent_to_operation(state, player, res.chosen_intent)
                if op is not None:
                    state.apply_operation_list(player, [op])

    for _ in range(max_rounds):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            if player == our_player:
                our_play(player)
            else:
                opp_play(player)
            if player == 1:
                state.advance_round()

    if state.terminal and state.winner is not None:
        score = 1.0 if state.winner == our_player else (0.0 if state.winner == opp_player else 0.5)
    else:
        hp_us = state.bases[our_player].hp
        hp_opp = state.bases[opp_player].hp
        score = 0.5 if hp_us == hp_opp else (1.0 if hp_us > hp_opp else 0.0)
    tag = {1.0: "WIN", 0.5: "DRAW", 0.0: "LOSS"}[score]
    side = "1st" if our_player == 0 else "2nd"
    print(f"  [{side}] seed={seed:3d} {tag} us={state.bases[our_player].hp:2d} "
          f"opp={state.bases[opp_player].hp:2d} rounds={state.round_index}", flush=True)
    return {"score": score}


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval intent-space AlphaZero vs reference opponent")
    parser.add_argument("--checkpoint", type=str, default=None, help="trained AZ checkpoint (.pt)")
    parser.add_argument("--baseline-hotstart", type=str, default=None, help="ES checkpoint to eval as baseline")
    parser.add_argument("--opponent", type=str, default="rule_v4", choices=["rule_v4", "example"])
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--max-depth-rounds", type=int, default=2)
    parser.add_argument("--no-search", action="store_true", help="use raw argmax policy (no MCTS)")
    parser.add_argument("--select-by-prior", action="store_true", help="search but pick argmax prior")
    parser.add_argument("--intent-decoding", action="store_true", default=True,
                        help="intent decoding (auto-downgrade for weapons) — default True")
    parser.add_argument("--one-head", action="store_true", help="decode only head1 (1 op/turn)")
    parser.add_argument("--bundle-mcts", action="store_true", help="use bundle-sampling MCTS")
    parser.add_argument("--k", type=int, default=24, help="bundle sampling top-k (default 24)")
    parser.add_argument("--c-puct", type=float, default=1.25,
                        help="MCTS exploration constant (explore = c_puct * prior * ...)")
    parser.add_argument("--t-class", type=float, default=1.0, help="bundle sampling class temperature")
    parser.add_argument("--t-pos", type=float, default=1.0, help="bundle sampling position temperature")
    parser.add_argument("--self-raw-opponent", action="store_true",
                        help="opponent = raw (no-search) decode of the same model")
    parser.add_argument("--max-rounds", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--our-player", type=int, default=-1,
                        help="fix our player role (0 or 1); default -1 = alternate by game index")
    parser.add_argument("--native-engine", action="store_true",
                        help="use the C++ engine (native_game) for the game simulation")
    args = parser.parse_args()

    if args.checkpoint is None and args.baseline_hotstart is None:
        parser.error("provide --checkpoint or --baseline-hotstart")
    src = args.checkpoint or args.baseline_hotstart

    jobs = [
        (args.checkpoint, args.iterations, args.max_depth_rounds, args.seed + s,
         args.our_player if args.our_player >= 0 else s % 2,
         args.opponent, args.baseline_hotstart, args.no_search, args.select_by_prior,
         args.intent_decoding, args.one_head, args.bundle_mcts, args.k, args.c_puct,
         args.t_class, args.t_pos,
         args.self_raw_opponent, args.max_rounds, args.native_engine)
        for s in range(args.games)
    ]
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
    label = "baseline-hotstart" if args.baseline_hotstart else Path(args.checkpoint).name
    print(f"\n=== {label} vs {args.opponent}: {win}W/{draw}D/{loss}L "
          f"(win_rate={win / len(scores):.1%}) ===")


if __name__ == "__main__":
    main()
