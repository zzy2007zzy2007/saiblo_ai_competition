"""Re-test gen_15 top1 vs ExampleAI (quick check)."""
import sys
sys.path = ['Ant-Game', 'code'] + sys.path
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import time, multiprocessing as mp


def _worker(seed):
    import os as _os
    _os.environ['OMP_NUM_THREADS'] = '1'
    _os.environ['MKL_NUM_THREADS'] = '1'
    import torch as _torch
    _torch.set_num_threads(1)
    from SDK.backend.engine import GameState
    from SDK.utils.constants import MAX_ROUND
    from AI.ai_example import AI as ExampleAI
    from my_ai.network import create_model
    from my_ai.agent import NeuralAgent

    ckpt = _torch.load('training_history/20260630_182516/gen_0015.pt', map_location='cpu', weights_only=True)
    vec = ckpt['top2_params'][0].numpy()
    model = create_model(single_head=True)
    model.set_parameters_from_vector(vec)
    agent = NeuralAgent(model=model)
    opp = ExampleAI(seed=seed)

    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    p = seed % 2
    for _ in range(MAX_ROUND):
        if state.terminal:
            break
        if p == 0:
            u, v = agent._choose_operations(state, 0), opp.choose_operations(state, 1)
        else:
            u, v = opp.choose_operations(state, 0), agent._choose_operations(state, 1)
        state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)

    h_me = state.bases[p].hp
    h_opp = state.bases[1 - p].hp
    score = 1.0 if h_me > h_opp else (0.5 if h_me == h_opp else 0.0)
    return score, h_me, h_opp


if __name__ == '__main__':
    t0 = time.time()
with mp.Pool(10) as pool:
    results = pool.map(_worker, range(10))

scores = [r[0] for r in results]
wins = sum(1 for s in scores if s == 1.0)
draws = sum(1 for s in scores if s == 0.5)
avg_h_me = sum(r[1] for r in results) / 10
avg_h_opp = sum(r[2] for r in results) / 10
print(f'gen_15 top1 vs ExampleAI:  {wins}/10 wins  {draws}/10 draws')
print(f'Avg HP: me={avg_h_me:.1f}  opp={avg_h_opp:.1f}  ({time.time()-t0:.0f}s)')
for s, (score, hm, ho) in enumerate(results):
    print(f'  seed {s}: {"WIN" if score==1 else "DRAW" if score==0.5 else "LOSS"}  me={hm}  opp={ho}')
