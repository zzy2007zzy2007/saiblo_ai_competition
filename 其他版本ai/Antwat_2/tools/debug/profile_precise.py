#!/usr/bin/env python3
"""
Per-step precise profiling: separate each phase, suppress log pollution.

Measures exact time of each sub-phase in a SelfPlay step:
  1. data_prep:   numpy → torch.tensor + .to(device)  [CPU alloc + PCIe DMA]
  2. gpu_forward: policy.get_action() + torch.cuda.synchronize()  [GPU compute]
  3. opp_move:    opponent mask prep + random action  [CPU only]
  4. env_step:    env._resolve_turn_mixed()  [CPU: game engine + obs encoding + reward]

Also answers:
  - Are GPU and CPU phases sequential or can they overlap?
  - How much does DEBUG logging inflate env_step time?
  - What is potential throughput with N envs + batch inference?

Usage:
    python tools/debug/profile_precise.py [--steps 500] [--device cuda] [--log-level WARNING]

Dependencies:
    export PYTHONPATH=ppo/src:/path/to/Ant-Game
"""

import argparse
import sys
import time
import statistics
import logging
from pathlib import Path

import numpy as np
import torch

# Suppress all DEBUG/INFO logs from SDK and ppo_antwar
logging.getLogger().setLevel(logging.WARNING)
for logger_name in ['SDK', 'ppo_antwar', '__main__']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PPO_SRC = REPO_ROOT / "ppo" / "src"
if str(PPO_SRC) not in sys.path:
    sys.path.insert(0, str(PPO_SRC))

# Try to resolve SDK path
from ppo_antwar.config.path_config import PathConfig
try:
    path_config = PathConfig()
    sdk_path = str(path_config.sdk_dir)
    sdk_parent = str(Path(sdk_path).parent)
    if sdk_parent not in sys.path:
        sys.path.insert(0, sdk_parent)
except Exception:
    pass

from loguru import logger
logger.remove()  # remove default handler
logger.add(lambda msg: None, level="WARNING")  # suppress all loguru output during measurement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000, help="Steps to profile")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    args = parser.parse_args()

    from ppo_antwar.env.antwar_env import AntWarEnv
    from ppo_antwar.utils import get_config

    config = get_config()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Steps: {args.steps}")

    # --- Create network (random weights, same as PPOTrainer init) ---
    from ppo_antwar.network.antwar_net import AntWarPolicyValueNetwork
    policy = AntWarPolicyValueNetwork(
        board_shape=tuple(config.network.board_shape),
        global_dim=config.network.global_dim,
        action_dim=config.network.action_dim,
        hidden_dim=config.network.hidden_dim,
    ).to(device)
    policy.eval()

    # Warm up GPU
    dummy_board = torch.randn(1, 28, 19, 19).to(device)
    dummy_global = torch.randn(1, 33).to(device)
    dummy_mask = torch.ones(1, 96).to(device)
    for _ in range(10):
        with torch.no_grad():
            policy.get_action(dummy_board, dummy_global, dummy_mask)
    if device == "cuda":
        torch.cuda.synchronize()

    # --- Create environment ---
    env = AntWarEnv()

    # --- Phase timing accumulators ---
    data_prep_times = []   # numpy → tensor → .to(device)
    gpu_fwd_times = []     # policy.get_action() + synchronize
    opp_move_times = []    # opponent mask + random choice
    env_step_times = []    # env._resolve_turn_mixed()
    step_total_times = []  # total wall time from start of GPU to end of env step

    obs, _ = env.reset()
    step = 0

    print(f"\nCollecting {args.steps} steps...")
    print("(logs suppressed to WARNING level, no DEBUG pollution)")

    while step < args.steps:
        player_obs = obs["player_0"]

        # ============ Phase 1: Data Preparation (CPU→GPU transfer) ============
        t0 = time.perf_counter()

        board = torch.FloatTensor(player_obs['board']).unsqueeze(0).to(device)
        global_obs = torch.FloatTensor(player_obs['global']).unsqueeze(0).to(device)
        action_mask = torch.FloatTensor(player_obs['action_mask']).unsqueeze(0).to(device)

        t1 = time.perf_counter()
        data_prep_times.append((t1 - t0) * 1000)

        # ============ Phase 2: GPU forward + synchronize ============
        with torch.no_grad():
            action, log_prob, value = policy.get_action(
                board, global_obs, action_mask, deterministic=False
            )

        if device == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        gpu_fwd_times.append((t2 - t1) * 1000)

        action_id = action.cpu().item()

        # ============ Phase 3: Opponent move (CPU only) ============
        opponent_mask = torch.FloatTensor(obs["player_1"]["action_mask"]).to(device)
        valid_actions = torch.where(opponent_mask > 0)[0]
        opp_action_id = int(np.random.choice(valid_actions.cpu().numpy())) if len(valid_actions) > 0 else 0

        t3 = time.perf_counter()
        opp_move_times.append((t3 - t2) * 1000)

        # ============ Phase 4: Env step (CPU: game engine) ============
        obs, rewards, terminated, truncated, info = env._resolve_turn_mixed(
            player_0_action_id=action_id,
            player_1_action_id=opp_action_id,
        )

        t4 = time.perf_counter()
        env_step_times.append((t4 - t3) * 1000)

        # Total: from start of data prep to end of env step
        step_total_times.append((t4 - t0) * 1000)

        if terminated or truncated:
            obs, _ = env.reset()

        step += 1
        if step % 200 == 0:
            print(f"  {step}/{args.steps}...")

    # ============ Print Results ============
    def stats(data):
        if len(data) < 2:
            return "insufficient"
        s = sorted(data)
        return (f"  mean={statistics.mean(data):>8.3f}  "
                f"median={statistics.median(data):>8.3f}  "
                f"p95={s[int(len(s)*0.95)]:>8.3f}  "
                f"min={min(data):>8.3f}  max={max(data):>8.3f}")

    print("\n" + "=" * 75)
    print(f"  PHASE                    mean     median   p95      min      max")
    print(f"  " + "-" * 66)
    print(f"  {'1. data_prep (CPU→GPU)':30s} {stats(data_prep_times)}")
    print(f"  {'2. gpu_forward (compute)':30s} {stats(gpu_fwd_times)}")
    print(f"  {'3. opp_move (CPU)':30s} {stats(opp_move_times)}")
    print(f"  {'4. env_step (CPU)':30s} {stats(env_step_times)}")
    print(f"  " + "-" * 66)
    print(f"  {'step_total (1+2+3+4)':30s} {stats(step_total_times)}")
    print(f"  " + "-" * 66)

    # Aggregation
    m1 = statistics.median(data_prep_times)
    m2 = statistics.median(gpu_fwd_times)
    m3 = statistics.median(opp_move_times)
    m4 = statistics.median(env_step_times)
    total = m1 + m2 + m3 + m4

    print(f"\n  Median phase distribution:")
    print(f"    data_prep:  {m1:>8.3f} ms ({m1/total*100:>5.1f}%)")
    print(f"    gpu_fwd:    {m2:>8.3f} ms ({m2/total*100:>5.1f}%)")
    print(f"    opp_move:   {m3:>8.3f} ms ({m3/total*100:>5.1f}%)")
    print(f"    env_step:   {m4:>8.3f} ms ({m4/total*100:>5.1f}%)")
    print(f"    total:      {total:>8.3f} ms")

    # ============ Core Question: Can GPU and CPU overlap? ============
    print("\n" + "=" * 75)
    print("  PIPELINE ANALYSIS: Can GPU and CPU work overlap?")
    print(f"  " + "-" * 66)
    print(f"  Currently: sequential per step")
    print(f"    GPU phase (prep+forward): {m1+m2:.2f}ms  |  CPU phases (opp+env): {m3+m4:.2f}ms")
    print(f"    → GPU idle {m3+m4:.2f}ms per step waiting for CPU")
    print(f"    → CPU idle {m1+m2:.2f}ms per step waiting for GPU")
    print(f"  ")
    print(f"  With async pipeline (next step's GPU overlaps with current step's CPU):")
    print(f"    bottleneck = max(GPU_time, CPU_time) = max({m1+m2:.2f}, {m3+m4:.2f}) = {max(m1+m2, m3+m4):.2f}ms")
    print(f"    theoretical speedup: {(m1+m2+m3+m4)/max(m1+m2, m3+m4):.2f}x")
    print(f"  ")
    print(f"  Note: data_prep includes CPU allocation which CANNOT overlap with env_step")
    print(f"  (both need CPU). Only gpu_fwd can overlap with CPU phases.")

    # ============ Multi-env Estimation ============
    print("\n" + "=" * 75)
    print("  MULTI-ENV THROUGHPUT ESTIMATION")
    print(f"  " + "-" * 66)

    total_cpu = m3 + m4
    total_gpu = m1 + m2

    print(f"  Single env: total={total:.2f}ms/step, 1 traj/{total*250/1000:.2f}s = {1/(total*250/1000):.2f} traj/s")
    print(f"")

    for N in [2, 4, 6, 8]:
        # Sequential: N envs, no parallelism
        seq_gpu = m1 * N ** 0.15 + m2 * N ** 0.1  # slight batch benefit
        seq_cpu = (m3 + m4) * N
        seq_total = seq_gpu + seq_cpu
        seq_throughput = N / (seq_total * 250 / 1000)

        # Threaded: N envs, env steps parallel, GPU batched
        thr_gpu = m1 * N ** 0.3 + m2 * N ** 0.15  # batch inference benefit
        thr_cpu = m3 * N ** 0.1 + m4 * N ** 0.1   # opponent moves also batched
        # With threading, CPU time is max(env_step) not sum(env_step)
        # but there's still overhead from thread management
        # Realistic: ~max(opp+env) + some contention (Python GIL)
        thr_cpu_parallel = (m3 + m4) * (1 + 0.15 * (N - 1))  # 15% overhead per extra env
        thr_total = thr_gpu + thr_cpu_parallel
        thr_throughput = N / (thr_total * 250 / 1000)

        # Overlapped (async): GPU and CPU pipelined
        ol_total = max(thr_gpu, thr_cpu_parallel) * (1 + 0.05 * (N - 1))
        ol_throughput = N / (ol_total * 250 / 1000)

        print(f"  N={N}:  ", end="")
        print(f"  seq={seq_throughput:>5.2f} traj/s  ", end="")
        print(f"|  thr={thr_throughput:>5.2f} traj/s  ", end="")
        print(f"|  overlap={ol_throughput:>5.2f} traj/s")

    print(f"  " + "-" * 66)
    print(f"  seq=sequential stepping  |  thr=threaded stepping  |  overlap=threaded+async GPU")
    print(f"  Current baseline: 1 env = {1/(total*250/1000):.2f} traj/s")
    print("=" * 75)


if __name__ == "__main__":
    main()
