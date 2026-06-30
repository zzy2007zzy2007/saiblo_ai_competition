"""Head-to-head comparison of two checkpoints."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2] / "Ant-Game"
_CODE = Path(__file__).resolve().parents[1]
for p in (_REPO, _CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import multiprocessing as mp
import time

from my_ai.network import create_model
from my_ai.agent import NeuralAgent


def _worker(args):
    s, ckpt_a, ckpt_b = args
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND

    ma = create_model(single_head=True)
    mb = create_model(single_head=True)
    ca = torch.load(ckpt_a, map_location="cpu", weights_only=True)
    cb = torch.load(ckpt_b, map_location="cpu", weights_only=True)
    va = ca["top2_params"][0].numpy()
    vb = cb["top2_params"][0].numpy()
    if ma.count_parameters() != len(va):
        ma = create_model(single_head=False)
    if mb.count_parameters() != len(vb):
        mb = create_model(single_head=False)
    ma.set_parameters_from_vector(va)
    mb.set_parameters_from_vector(vb)
    aa = NeuralAgent(model=ma)
    ab = NeuralAgent(model=mb)

    state = GameState.initial(seed=s, cold_handle_rule_illegal=True)
    p = s % 2
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        u = aa._choose_operations(state, p) if p == 0 else ab._choose_operations(state, p)
        v = ab._choose_operations(state, 1 - p) if p == 0 else aa._choose_operations(state, 1 - p)
        state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)

    h0 = state.bases[0].hp
    h1 = state.bases[1].hp
    if h0 > h1:
        return 1.0
    elif h1 > h0:
        return 0.0
    return 0.5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_a", type=str)
    parser.add_argument("ckpt_b", type=str)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    with mp.Pool(args.workers) as pool:
        scores = pool.map(_worker, [(s, args.ckpt_a, args.ckpt_b) for s in range(args.games)])
    dt = time.perf_counter() - t0

    wins = sum(1 for s in scores if s == 1.0)
    draws = sum(1 for s in scores if s == 0.5)
    losses = sum(1 for s in scores if s == 0.0)
    a_name = Path(args.ckpt_a).stem
    b_name = Path(args.ckpt_b).stem
    print(f"\n{a_name} vs {b_name}  ({args.games} games, {args.workers} workers, {dt:.0f}s)")
    print(f"  {a_name} win rate: {wins}/{args.games} ({100*wins/args.games:.1f}%)")
    print(f"  Draws: {draws}/{args.games}")
    print(f"  {b_name} win rate: {losses}/{args.games} ({100*losses/args.games:.1f}%)")
