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
    ckpt_path, iterations, max_depth_rounds, seed, our_player, opponent, hotstart, no_search, select_by_prior, intent_decoding, one_head, bundle_mcts, depth0, value_tanh, value_rel_to_abs, k, c_puct, t_class, t_pos, self_raw_opponent, max_rounds, native_engine, random_action_prob, search_mode, pos_pin = args
    import torch
    torch.set_num_threads(1)

    from my_ai.az_intent.az_selfplay import make_initial_state
    from SDK.utils.features import FeatureExtractor

    if hotstart:
        from my_ai.az_intent.train import build_model
        model = build_model(hotstart)
    else:
        from my_ai.network import create_model, model_kwargs_from_ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = create_model(**model_kwargs_from_ckpt(ckpt))
        model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat = FeatureExtractor(max_actions=96)
    state = make_initial_state(seed, native_engine)
    opp_player = 1 - our_player
    diag = None  # per-turn class/position diagnostics (bundle-MCTS path only)

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

        pol_model, net_fn = make_net_fn_from_ckpt(ckpt_path, feat, value_tanh=value_tanh,
                                                   value_rel_to_abs=value_rel_to_abs)
        if depth0:
            from my_ai.az_intent.az_depth0_greedy import Depth0Greedy
            bmcts = Depth0Greedy(net_fn, iterations=iterations, k=k,
                                 c_puct=c_puct, t_class=t_class, t_pos=t_pos, seed=seed)
        else:
            bmcts = BundleMCTS(net_fn, iterations=iterations, max_depth_rounds=max_depth_rounds,
                               k=k, c_puct=c_puct, t_class=t_class, t_pos=t_pos, seed=seed,
                               search_mode=search_mode, pos_pin=pos_pin)
        ra_rng = np.random.default_rng(seed + 7777)

        # Per-turn diagnostics for the class/position ablation: how often does the
        # search even have room to differ, and when it does, does the difference
        # land on the CLASS axis or the POSITION axis?
        diag = {"turns": 0, "room": 0, "same": 0, "class": 0, "pos": 0, "both": 0}

        def _raw_bundle_and_intents(pl):
            from my_ai.decoder import decode_network_output
            obs = feat.encode_observation(state, pl, np.zeros(96))
            with torch.no_grad():
                out = pol_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            sc, sp = [], []
            ops = decode_network_output(out, state, pl, temperature=0.0,
                                       intent_decoding=True,
                                       sampled_class_out=sc, sampled_pos_out=sp)
            pos_by_ch = {int(t[0]): (int(t[1]), int(t[2])) for t in sp}
            intents = [(int(c), *pos_by_ch.get(int(c), (-1, -1))) for c in sc]
            return ops, intents

        def our_play(player):
            res = bmcts.search(state, player, temperature=0.0)
            chosen = res.chosen_bundle
            diag["turns"] += 1
            if len(set(res.bundles)) >= 2:
                diag["room"] += 1
            raw_ops, raw_int = _raw_bundle_and_intents(player)
            if tuple((int(k[0]), int(k[1]), int(k[2])) for k in chosen) == \
                    tuple((int(o.op_type), o.arg0, o.arg1) for o in raw_ops):
                diag["same"] += 1
            else:
                ic = res.intent_counts[res.chosen_index] if res.intent_counts else []
                chosen_int = [max(d, key=d.get) if d else (-1, -1, -1) for d in ic]
                c_d, p_d = False, False
                for h in range(min(len(chosen_int), len(raw_int))):
                    c1, x1, y1 = chosen_int[h]
                    c2, x2, y2 = raw_int[h]
                    if c1 < 0 or c2 < 0:
                        continue
                    if c1 != c2:
                        c_d = True
                    elif (x1, y1) != (x2, y2):
                        p_d = True
                if c_d and p_d:
                    diag["both"] += 1
                elif c_d:
                    diag["class"] += 1
                elif p_d:
                    diag["pos"] += 1
            if random_action_prob > 0.0 and ra_rng.random() < random_action_prob:
                from my_ai.az_intent.az_selfplay import _random_legal_bundle
                from my_ai.decoder import make_position_masks, make_class_mask
                obs = feat.encode_observation(state, player, np.zeros(96))
                with torch.no_grad():
                    out = pol_model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
                pm_ = make_position_masks(state, player, intent_decoding=True)
                cm_ = make_class_mask(state, player, position_mask=pm_, intent_decoding=True)
                rb = _random_legal_bundle(pol_model, out, state, player, ra_rng, pm_, cm_)
                if rb:
                    chosen = rb
            ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in chosen
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
    if diag is not None and diag["turns"]:
        print(f"      diag turns={diag['turns']} room(>=2 cands)={diag['room'] / diag['turns']:.1%} "
              f"same-as-raw={diag['same'] / diag['turns']:.1%} "
              f"diff[class={diag['class']} pos={diag['pos']} both={diag['both']}]", flush=True)
    return {"score": score, "diag": diag}


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
    parser.add_argument("--depth0", action="store_true",
                        help="use depth-0 greedy search (value candidate post-apply states) "
                             "instead of full MCTS rollout")
    parser.add_argument("--no-value-tanh", action="store_true",
                        help="keep the raw value-head output (skip tanh) in net_fn — "
                             "experiment: tanh saturates large abs-label value heads")
    parser.add_argument("--value-rel-to-abs", type=float, default=0.0,
                        help="0=off.  For a value head trained on RELATIVE labels"
                             " (label=(future_hp-d_t)/20*scale), reconstruct the ABSOLUTE"
                             " value as raw/scale + d_t/HP_SCALE.  Use with --no-value-tanh.")
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
    parser.add_argument("--random-action-prob", type=float, default=0.0,
                        help="x: with probability x OUR play ignores the search and plays a "
                             "uniformly random legal bundle (tests what injecting randomness costs)")
    parser.add_argument("--search-mode", type=str, default="joint",
                        choices=["joint", "class-only", "pos-only"],
                        help="bundle-MCTS candidate axis ablation: joint = sample classes AND "
                             "positions (default); pos-only = pin the class and vary only "
                             "positions; class-only = pin positions to argmax and vary only "
                             "classes.  Pair with --self-raw-opponent to see which axis the "
                             "search actually contributes on.")
    parser.add_argument("--pos-pin", type=str, default="argmax",
                        choices=["argmax", "playable"],
                        help="pos-only mode: which class to pin.  argmax = the raw class "
                             "argmax (literal reading; collapses to raw whenever that class "
                             "is HOLD/unaffordable); playable = highest-logit class that "
                             "actually executes, which keeps the position axis searchable.")
    args = parser.parse_args()

    if args.checkpoint is None and args.baseline_hotstart is None:
        parser.error("provide --checkpoint or --baseline-hotstart")
    src = args.checkpoint or args.baseline_hotstart

    jobs = [
        (args.checkpoint, args.iterations, args.max_depth_rounds, args.seed + s,
         args.our_player if args.our_player >= 0 else s % 2,
         args.opponent, args.baseline_hotstart, args.no_search, args.select_by_prior,
         args.intent_decoding, args.one_head, args.bundle_mcts, args.depth0,
         not args.no_value_tanh, args.value_rel_to_abs, args.k, args.c_puct,
         args.t_class, args.t_pos,
         args.self_raw_opponent, args.max_rounds, args.native_engine,
         args.random_action_prob, args.search_mode, args.pos_pin)
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
    opp_label = "raw-self" if args.self_raw_opponent else args.opponent
    mode_tag = "" if args.search_mode == "joint" else f" [search_mode={args.search_mode}]"
    if args.search_mode == "pos-only" and args.pos_pin != "argmax":
        mode_tag += f" [pos_pin={args.pos_pin}]"
    print(f"\n=== {label} vs {opp_label}{mode_tag}: {win}W/{draw}D/{loss}L "
          f"(win_rate={win / len(scores):.1%}) ===")

    agg = {k: 0 for k in ("turns", "room", "same", "class", "pos", "both")}
    for r in results:
        d = r.get("diag")
        if d:
            for k in agg:
                agg[k] += d[k]
    if agg["turns"]:
        t = agg["turns"]
        unattributed = 1.0 - (agg["same"] + agg["class"] + agg["pos"] + agg["both"]) / t
        print(f"[diag] our-turns={t}  room(>=2 candidates)={agg['room'] / t:.1%}  "
              f"chosen==raw={agg['same'] / t:.1%}  differs-in: "
              f"class={agg['class'] / t:.1%} pos={agg['pos'] / t:.1%} both={agg['both'] / t:.1%} "
              f"unattributed={unattributed:.1%}")


if __name__ == "__main__":
    main()
