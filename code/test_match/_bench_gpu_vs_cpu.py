"""Real match benchmark: CPU single vs CPU multi vs GPU single."""
import sys, time, multiprocessing as mp
from pathlib import Path
sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/code').resolve()))
sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/Ant-Game').resolve()))

import torch
import numpy as np
from my_ai.network import create_model
from my_ai.decoder import decode_network_output
from SDK.backend.engine import GameState
from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import MAX_ROUND

SEED = 7
MAX_R = 120  # run 120 rounds per match for faster testing

def run_match_cpu_single(model, device='cpu'):
    fe = FeatureExtractor(max_actions=96)
    state = GameState.initial(seed=SEED, cold_handle_rule_illegal=True)
    model = model.to('cpu')
    model.eval()

    for r in range(MAX_R):
        if state.terminal: break
        for player in (0, 1):
            obs = fe.encode_observation(state, player, np.zeros(96))
            board = torch.from_numpy(obs['board']).unsqueeze(0).float()
            stats = torch.from_numpy(obs['stats']).unsqueeze(0).float()
            with torch.no_grad():
                out = model(board, stats)
            output = {
                'action_map': out['action_map'],
                'head1_logits': out['head1_logits'],
                'head2_logits': out['head2_logits'],
                'head3_logits': out['head3_logits'],
                'value': out['value'],
            }
            ops = decode_network_output(output, state, player)
            state.apply_operation_list(player, ops)
        state.advance_round()
    return r + 1

def run_match_gpu_single(model):
    device = torch.device('cuda')
    fe = FeatureExtractor(max_actions=96)
    state = GameState.initial(seed=SEED, cold_handle_rule_illegal=True)
    model = model.to(device)
    model.eval()

    for r in range(MAX_R):
        if state.terminal: break
        for player in (0, 1):
            obs = fe.encode_observation(state, player, np.zeros(96))
            board = torch.from_numpy(obs['board']).unsqueeze(0).float().to(device, non_blocking=True)
            stats = torch.from_numpy(obs['stats']).unsqueeze(0).float().to(device, non_blocking=True)
            with torch.no_grad():
                out = model(board, stats)
            # 移到 CPU 再解码
            output = {k: v.cpu() for k, v in out.items()}
            ops = decode_network_output(output, state, player)
            state.apply_operation_list(player, ops)
        state.advance_round()
    torch.cuda.synchronize()
    return r + 1

def _worker_cpu(seed):
    """Worker for multiprocessing: each process creates its own model."""
    import sys
    sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/code').resolve()))
    sys.path.insert(0, str(Path('d:/2026智能体大赛_新2/Ant-Game').resolve()))
    from my_ai.network import create_model
    fe = FeatureExtractor(max_actions=96)
    model = create_model().eval()
    state = GameState.initial(seed=seed, cold_handle_rule_illegal=True)
    for r in range(MAX_R):
        if state.terminal: break
        for player in (0, 1):
            obs = fe.encode_observation(state, player, np.zeros(96))
            board = torch.from_numpy(obs['board']).unsqueeze(0).float()
            stats = torch.from_numpy(obs['stats']).unsqueeze(0).float()
            with torch.no_grad():
                out = model(board, stats)
            ops = decode_network_output(out, state, player)
            state.apply_operation_list(player, ops)
        state.advance_round()
    return r + 1

# ── Run benchmarks ──
print("=" * 55)
print("真实对战性能对比 (NeuralAgent vs NeuralAgent, 120回合)")
print("=" * 55)

model = create_model()

# 1. CPU 单进程
t0 = time.perf_counter()
r = run_match_cpu_single(model)
t_cpu = time.perf_counter() - t0
print(f"\nCPU 单进程:       {r}回合, {t_cpu:.2f}s, {t_cpu/r*1000:.0f}ms/回合")

# 2. GPU 单进程
t0 = time.perf_counter()
r = run_match_gpu_single(model)
torch.cuda.synchronize()
t_gpu = time.perf_counter() - t0
print(f"GPU 单进程:       {r}回合, {t_gpu:.2f}s, {t_gpu/r*1000:.0f}ms/回合")
print(f"  加速比 vs CPU:  {t_cpu/t_gpu:.1f}x")

# 3. CPU 多进程 (4 workers)
N_WORKERS = 4
t0 = time.perf_counter()
with mp.Pool(N_WORKERS) as pool:
    results = pool.map(_worker_cpu, [SEED + i for i in range(N_WORKERS)])
t_mp = time.perf_counter() - t0
total_rounds = sum(results)
print(f"CPU {N_WORKERS}进程并行: {N_WORKERS}场, {t_mp:.2f}s, {t_mp/N_WORKERS:.2f}s/场")
print(f"  GPU单进程等效吞吐: {t_gpu/N_WORKERS:.2f}s/场  (同时跑{N_WORKERS}场的均摊)")
print(f"  加速比 vs CPU多进程: {t_mp/t_gpu:.1f}x")

# 4. CPU 多进程 (8 workers)
N_WORKERS2 = 8
t0 = time.perf_counter()
with mp.Pool(N_WORKERS2) as pool:
    results = pool.map(_worker_cpu, [SEED + i for i in range(N_WORKERS2)])
t_mp2 = time.perf_counter() - t0
print(f"\nCPU {N_WORKERS2}进程并行: {N_WORKERS2}场, {t_mp2:.2f}s, {t_mp2/N_WORKERS2:.2f}s/场")
