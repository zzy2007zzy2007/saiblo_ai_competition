"""Diagnose why the MCTS search degrades the strong policy.

Plays one game vs ExampleAI with the SEARCH as our side, and per turn compares:
  - search_ops: the 3 ops chosen by 3 MCTS searches (one per head)
  - raw_ops:    the 3 ops the raw argmax policy would decode on the SAME state
At the FIRST divergent turn, dumps the search root internals (top intents by
prior, with visits and mean values) so we can see whether the search is being
misled by the prior, the value, or the visit selection.

Usage:
    python code/my_ai/az_intent/diagnose_search.py --checkpoint .../gen0120_warm.pt \
        --iterations 32 --max-depth-rounds 2 --seed 0
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


def describe_op(op) -> str:
    if op is None:
        return "HOLD"
    return f"op{int(op.op_type)}[{op.arg0},{op.arg1}]"


def intent_desc(it) -> str:
    return f"class{it.class_id}@({it.x},{it.y})" if it.class_id <= 20 else f"class{it.class_id}"


def main() -> None:
    from SDK.backend.engine import GameState
    from SDK.utils.features import FeatureExtractor
    from AI.ai_example import AI as ExampleAI
    from my_ai.az_intent.mcts import IntentMCTS, intent_to_operation
    from my_ai.az_intent.train import make_net_fn
    from my_ai.decoder import decode_network_output

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--max-depth-rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(4)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from my_ai.network import create_model
    model = create_model(
        num_resblocks=ckpt.get("num_resblocks", 6),
        num_heads=ckpt.get("num_heads", 3),
        latent_dim=ckpt.get("latent_dim", 64),
        no_bn=True,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat = FeatureExtractor(max_actions=96)
    net_fn = make_net_fn(model, feat)
    mcts = IntentMCTS(net_fn, iterations=args.iterations,
                      max_depth_rounds=args.max_depth_rounds, seed=args.seed)

    state = GameState.initial(seed=args.seed, cold_handle_rule_illegal=True)
    opp = ExampleAI(seed=args.seed)
    our_player = args.seed % 2
    opp_player = 1 - our_player

    divergent_turn = None
    print(f"our_player={our_player}  (search vs ExampleAI)\n")
    for round_idx in range(512):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            if player != our_player:
                ops = opp.choose_operations(state, player)
                state.apply_operation_list(player, ops)
                continue

            # ── raw policy ops on the same state ────────────────────────
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            raw_ops = decode_network_output(out, state, player,
                                            temperature=0.0, intent_decoding=True)

            # ── search ops (3 searches, one per head) ───────────────────
            search_ops = []
            for head in (0, 1, 2):
                if state.terminal:
                    break
                res = mcts.search(state, player, head_idx=head,
                                  temperature=0.0, add_root_noise=False)
                op = intent_to_operation(state, player, res.chosen_intent)
                search_ops.append(op)
                if op is not None:
                    state.apply_operation_list(player, [op])

            raw_desc = " + ".join(describe_op(o) for o in raw_ops)
            search_desc = " + ".join(describe_op(o) for o in search_ops)
            same = raw_desc == search_desc
            mark = "  " if same else "!!"
            print(f"  r{round_idx:3d} p{player}  raw={raw_desc}")
            print(f"       {mark} search={search_desc}  {'' if same else '<-- DIFFERS'}")
            if not same and divergent_turn is None:
                divergent_turn = (round_idx, player)

        if player == 1 and not state.terminal:
            state.advance_round()

    print(f"\nfinal: winner={state.winner} hp={[b.hp for b in state.bases]} "
          f"rounds={state.round_index}")
    print(f"first divergent turn: {divergent_turn}")
    print(f"totally divergent turns: "
          f"{(round_idx, player) if divergent_turn else 'none'}")

    # ── Re-run the search at the divergent turn and dump root internals ──
    if divergent_turn is None:
        print("\nNo divergence found (search matches raw policy) — game loop issue?")
        return

    print(f"\n=== re-running search at divergent turn {divergent_turn} ===")
    state2 = GameState.initial(seed=args.seed, cold_handle_rule_illegal=True)
    opp2 = ExampleAI(seed=args.seed)
    target_round, target_player = divergent_turn
    done = False
    for round_idx in range(512):
        if done or state2.terminal:
            break
        for player in (0, 1):
            if done or state2.terminal:
                break
            if player == target_player and round_idx == target_round:
                res = mcts.search(state2, player, head_idx=0,
                                  temperature=0.0, add_root_noise=False)
                done = True
                break
            if player != our_player:
                ops = opp2.choose_operations(state2, player)
                state2.apply_operation_list(player, ops)
            else:
                for head in (0, 1, 2):
                    if state2.terminal:
                        break
                    r = mcts.search(state2, player, head_idx=head,
                                    temperature=0.0, add_root_noise=False)
                    op = intent_to_operation(state2, player, r.chosen_intent)
                    if op is not None:
                        state2.apply_operation_list(player, [op])
        if player == 1 and not state2.terminal:
            state2.advance_round()

    root = mcts.last_root
    print(f"\nsearch chose: {intent_desc(res.chosen_intent)} -> {describe_op(intent_to_operation(state2, target_player, res.chosen_intent))}")
    print(f"raw policy on this state:")
    obs = feat.encode_observation(state2, target_player, np.zeros(96))
    with torch.no_grad():
        out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                    torch.from_numpy(obs["stats"]).unsqueeze(0).float())
    raw_ops = decode_network_output(out, state2, target_player, temperature=0.0, intent_decoding=True)
    print(f"  raw ops: {' + '.join(describe_op(o) for o in raw_ops)}")

    print(f"\nroot top-12 intents by prior  (head0):")
    print(f"  {'intent':22s} {'prior':>7s} {'visits':>6s} {'mean_val':>9s}")
    order = np.argsort(root.child_priors)[::-1]
    for i in order[:12]:
        it = root.intents[i]
        child = root.children[i]
        print(f"  {intent_desc(it):22s} {root.child_priors[i]:7.4f} "
              f"{child.visits:6d} {child.mean_value:9.3f}")


if __name__ == "__main__":
    main()
