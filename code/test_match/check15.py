"""Quick re-test gen_15 top1 vs ExampleAI (Windows-safe)."""
import sys, os, time, multiprocessing as mp
sys.path = ['Ant-Game', 'code'] + sys.path
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch; torch.set_num_threads(1)

def _worker(seed):
    import os as _o; _o.environ['OMP_NUM_THREADS'] = '1'; _o.environ['MKL_NUM_THREADS'] = '1'
    import torch as _t; _t.set_num_threads(1)
    from SDK.backend.engine import GameState; from SDK.utils.constants import MAX_ROUND
    from AI.ai_example import AI as ExampleAI
    from my_ai.network import create_model; from my_ai.agent import NeuralAgent
    ckpt = _t.load('training_history/20260630_182516/gen_0015.pt', map_location='cpu', weights_only=True)
    model = create_model(single_head=True)
    model.set_parameters_from_vector(ckpt['top2_params'][0].numpy())
    agent = NeuralAgent(model=model); opp = ExampleAI(seed=seed)
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    p = seed % 2
    for _ in range(MAX_ROUND):
        if state.terminal: break
        if p == 0: u, v = agent._choose_operations(state, 0), opp.choose_operations(state, 1)
        else: u, v = opp.choose_operations(state, 0), agent._choose_operations(state, 1)
        state.resolve_turn(u, v) if p == 0 else state.resolve_turn(v, u)
    h = state.bases[p].hp; h2 = state.bases[1-p].hp
    return 1.0 if h > h2 else (0.5 if h == h2 else 0.0), h, h2

if __name__ == '__main__':
    t0 = time.time()
    with mp.Pool(10) as pool:
        results = pool.map(_worker, range(10))
    w = sum(1 for r in results if r[0]==1.0); d = sum(1 for r in results if r[0]==0.5)
    avg_hm = sum(r[1] for r in results)/10; avg_ho = sum(r[2] for r in results)/10
    print(f'gen_15 vs ExampleAI: {w}/10 wins {d}/10 draws ({time.time()-t0:.0f}s)')
    print(f'Avg HP: me={avg_hm:.1f}  opp={avg_ho:.1f}')
    for i, (s, hm, ho) in enumerate(results):
        print(f'  {i}: {"WIN" if s==1 else "DRAW" if s==0.5 else "LOSS"}  me={hm}  opp={ho}')
