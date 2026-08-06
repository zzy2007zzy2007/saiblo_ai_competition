"""Diagnose bundle-MCTS vs the raw policy: where do they diverge, and why.

Plays one game vs ExampleAI with the bundle MCTS; per turn compares the search's
chosen bundle with the raw argmax policy's decode on the same state.  At the
first divergent turn, dumps the search root (top bundles by prior/visits/value)
to see whether the search is exploring a worse alternative or missing the raw
policy's move.

Usage:
    python code/my_ai/az_intent/diagnose_bundle.py --checkpoint .../gen0120_warm.pt \
        --iterations 32 --seed 0 [--t-class 1.0 --t-pos 1.0]
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


def describe_ops(ops) -> str:
    if not ops:
        return "WAIT"
    return " + ".join(f"op{int(o.op_type)}[{o.arg0},{o.arg1}]" for o in ops)


def bundle_desc(key: tuple) -> str:
    if not key:
        return "WAIT"
    return " + ".join(f"op{k[0]}[{k[1]},{k[2]}]" for k in key)


def main() -> None:
    from SDK.backend.engine import GameState
    from SDK.utils.features import FeatureExtractor
    from AI.ai_example import AI as ExampleAI
    from SDK.backend.model import Operation
    from SDK.utils.constants import OperationType
    from my_ai.az_intent.bundle_mcts import BundleMCTS
    from my_ai.az_intent.train import make_net_fn
    from my_ai.decoder import decode_network_output

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--max-depth-rounds", type=int, default=2)
    parser.add_argument("--t-class", type=float, default=1.0)
    parser.add_argument("--t-pos", type=float, default=1.0)
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
    mcts = BundleMCTS(net_fn, iterations=args.iterations,
                      max_depth_rounds=args.max_depth_rounds,
                      t_class=args.t_class, t_pos=args.t_pos, seed=args.seed)

    state = GameState.initial(seed=args.seed, cold_handle_rule_illegal=True)
    opp = ExampleAI(seed=args.seed)
    our_player = args.seed % 2

    divergences = []
    print(f"our_player={our_player} (bundle MCTS vs ExampleAI, t={args.t_class}/{args.t_pos})\n")
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

            # raw policy decode on the same state
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            raw_ops = decode_network_output(out, state, player,
                                            temperature=0.0, intent_decoding=True)

            # search's chosen bundle
            res = mcts.search(state, player, temperature=0.0)
            search_ops = [
                Operation(OperationType(int(k[0])), int(k[1]), int(k[2]))
                for k in res.chosen_bundle
            ]
            state.apply_operation_list(player, search_ops)

            same = describe_ops(search_ops) == describe_ops(raw_ops)
            mark = "  " if same else "!!"
            print(f"  r{round_idx:3d} p{player}  raw={describe_ops(raw_ops)}")
            print(f"       {mark} search={describe_ops(search_ops)}  {'' if same else '<-- DIFFERS'}")
            if not same:
                divergences.append((round_idx, player, describe_ops(raw_ops), describe_ops(search_ops)))

        if player == 1 and not state.terminal:
            state.advance_round()

    print(f"\nfinal: winner={state.winner} hp={[b.hp for b in state.bases]} rounds={state.round_index}")
    print(f"divergent turns: {len(divergences)} / total our-turns ~{round_idx+1}")
    print(f"first 5 divergences: {divergences[:5]}")
    print(f"last 5 divergences: {divergences[-5:]}")


if __name__ == "__main__":
    main()
