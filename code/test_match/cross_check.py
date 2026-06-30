"""Cross-check: gen_0010 vs gen_0013/0014/0015 (multi-process)."""
from __future__ import annotations
import sys, os, time, multiprocessing as mp
sys.path = ['Ant-Game', 'code'] + sys.path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

BASE = 'training_history/20260630_182516'


def _worker(args):
    seed, ckpt0_path, ckpt1_path = args
    import os as _os
    _os.environ['OMP_NUM_THREADS'] = '1'
    _os.environ['MKL_NUM_THREADS'] = '1'
    import torch as _torch
    _torch.set_num_threads(1)
    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    m0 = create_model(single_head=True)
    m1 = create_model(single_head=True)

    ckpt0 = _torch.load(ckpt0_path, map_location='cpu', weights_only=True)
    ckpt1 = _torch.load(ckpt1_path, map_location='cpu', weights_only=True)

    # Use top1 from each
    vec0 = ckpt0.get('top2_params', [ckpt0['mean']])[0]
    vec1 = ckpt1.get('top2_params', [ckpt1['mean']])[0]
    m0.set_parameters_from_vector(vec0.numpy())
    m1.set_parameters_from_vector(vec1.numpy())

    a0 = NeuralAgent(model=m0)
    a1 = NeuralAgent(model=m1)

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    p = seed % 2
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        if p == 0:
            u, v = a0._choose_operations(state, 0), a1._choose_operations(state, 1)
        else:
            u, v = a1._choose_operations(state, 0), a0._choose_operations(state, 1)
        state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)

    h0, h1 = state.bases[0].hp, state.bases[1].hp
    return 1.0 if h0 > h1 else (0.5 if h0 == h1 else 0.0)


def main():
    pairs = [(10, 15), (10, 14), (10, 13)]
    for g_old, g_new in pairs:
        t0 = time.time()
        with mp.Pool(12) as pool:
            scores = pool.map(_worker, [
                (s, f'{BASE}/gen_{g_old:04d}.pt', f'{BASE}/gen_{g_new:04d}.pt')
                for s in range(30)
            ])
        wins = sum(1 for s in scores if s == 1.0)
        draws = sum(1 for s in scores if s == 0.5)
        print(f'gen_{g_old:02d} vs gen_{g_new:02d}:  {wins}/30 wins  {draws}/30 draws  ({time.time()-t0:.0f}s)')
        # Also compute which player won
        # Since we always have gen_old as player 0 on even seeds, gen_new as player 0 on odd seeds
        # The score is from player 0's perspective, so it's not directly "gen_old vs gen_new"
        # We need to check: who won?
        # Actually _worker returns 1.0 if player 0 wins. Player 0 could be either checkpoint.
        # For proper interpretation, let's track separately.
        # Skip detailed tracking for now — just see if there's any difference.
        w_old = sum(1 for s, i in zip(scores, range(30)) if (i % 2 == 0 and s == 1.0) or (i % 2 == 1 and s == 0.0))
        print(f'  → gen_{g_old} won: {w_old}/30  gen_{g_new} won: {30 - w_old - draws}/30')


if __name__ == '__main__':
    main()
