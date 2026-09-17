"""Measure wall-clock cost of BundleMCTS.search at different max_depth_rounds.

Question: does reducing depth save time?  The cost per iteration is dominated by
one `_expand` (net_fn + k*sample_mult bundle samples + up to k child state
clones with apply_operation_list/advance_round).  Depth only caps how deep the
tree may grow, so if 256 iterations never reach the cap, depth should not matter.

Usage:
    python code/my_ai/az_intent/_bench_depth_cost.py --checkpoint <ckpt> \
        [--iterations 256] [--depths 2 4] [--reps 3] [--advance-rounds 30]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[2]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--advance-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--t-class", type=float, default=0.5)
    parser.add_argument("--t-pos", type=float, default=1.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    from SDK.utils.features import FeatureExtractor
    from my_ai.decoder import decode_network_output
    from my_ai.az_intent.az_selfplay import make_initial_state, make_net_fn_from_ckpt
    from my_ai.az_intent.bundle_mcts import BundleMCTS

    feat = FeatureExtractor(max_actions=96)
    model, net_fn = make_net_fn_from_ckpt(args.checkpoint, feat)
    state = make_initial_state(args.seed, native_engine=True)

    # advance into the midgame with the raw policy so the state is non-trivial
    for _ in range(args.advance_rounds):
        if state.terminal:
            break
        for player in (0, 1):
            if state.terminal:
                break
            obs = feat.encode_observation(state, player, np.zeros(96))
            with torch.no_grad():
                out = model(torch.from_numpy(obs["board"]).unsqueeze(0).float(),
                            torch.from_numpy(obs["stats"]).unsqueeze(0).float())
            ops = decode_network_output(out, state, player, temperature=0.0,
                                        intent_decoding=True)
            state.apply_operation_list(player, ops)
        if not state.terminal:
            state.advance_round()

    player = 0
    print(f"[bench] checkpoint={args.checkpoint} state round={state.round_index} "
          f"terminal={state.terminal} iterations={args.iterations} k={args.k}", flush=True)
    for depth in args.depths:
        times = []
        n_children = []
        for rep in range(args.reps):
            bmcts = BundleMCTS(net_fn, iterations=args.iterations, max_depth_rounds=depth,
                               k=args.k, t_class=args.t_class, t_pos=args.t_pos,
                               seed=1000 + rep)
            t0 = time.perf_counter()
            res = bmcts.search(state, player, temperature=0.0)
            dt = time.perf_counter() - t0
            times.append(dt)
            n_children.append(len(res.bundles))
        med = sorted(times)[len(times) // 2]
        print(f"[bench] depth={depth}: median={med:.3f}s  all={[round(t, 3) for t in times]}  "
              f"root_bundles={n_children}", flush=True)


if __name__ == "__main__":
    main()
