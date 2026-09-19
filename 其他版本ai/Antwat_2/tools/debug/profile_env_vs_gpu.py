#!/usr/bin/env python3
"""
Profile the time distribution of SelfPlay single step:
  - env._resolve_turn_from_ops()  (CPU: game engine + observation encoding + reward)
  - trainer._select_action()      (GPU: network forward + action sampling)

Usage:
    python tools/debug/profile_env_vs_gpu.py [--steps 500] [--device cuda]

Output:
    Per-step timing stats (mean, median, p95, p99) for each phase.

Dependencies:
    - Must run on server9 (or wherever SDK is accessible)
    - Requires ppo/src/ppo_antwar in PYTHONPATH
"""

import argparse
import sys
import time
import statistics
from pathlib import Path

import numpy as np
import torch

# --- Setup PYTHONPATH -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PPO_SRC = REPO_ROOT / "ppo" / "src"
if str(PPO_SRC) not in sys.path:
    sys.path.insert(0, str(PPO_SRC))

# --- SDK path ----------------------------------------------------------
# Try to resolve SDK path, but don't fail if not found
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to profile")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    args = parser.parse_args()

    from ppo_antwar.env.antwar_env import AntWarEnv, make_antwar_env
    from ppo_antwar.utils import get_config

    logger.info(f"Loading config...")
    config = get_config()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    logger.info(f"Using device: {device}")

    # --- Create network (same as PPOTrainer) ---------------------------
    from ppo_antwar.network.antwar_net import AntWarPolicyValueNetwork
    policy = AntWarPolicyValueNetwork(
        board_shape=tuple(config.network.board_shape),
        global_dim=config.network.global_dim,
        action_dim=config.network.action_dim,
        hidden_dim=config.network.hidden_dim,
    ).to(device)
    policy.eval()

    # --- Create environment --------------------------------------------
    logger.info("Creating env...")
    env = AntWarEnv()

    # --- Collect timing data -------------------------------------------
    gpu_times = []
    env_times = []
    step_times = []   # total time per step

    obs, _ = env.reset()
    done = False
    step = 0

    # Opponent: random agent (simulate selfplay opponent)
    import random as py_random

    logger.info(f"Profiling {args.steps} steps...")

    while step < args.steps:
        # ---- 1. GPU inference: _select_action equivalent --------------
        player_obs = obs["player_0"]

        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()

        board = torch.FloatTensor(player_obs['board']).unsqueeze(0).to(device)
        global_obs = torch.FloatTensor(player_obs['global']).unsqueeze(0).to(device)
        action_mask = torch.FloatTensor(player_obs['action_mask']).unsqueeze(0).to(device)

        with torch.no_grad():
            action, log_prob, value = policy.get_action(
                board, global_obs, action_mask, deterministic=False
            )

        torch.cuda.synchronize() if device == "cuda" else None
        t1 = time.perf_counter()

        action_id = action.cpu().item()
        gpu_times.append((t1 - t0) * 1000)  # ms

        # ---- 2. Env step: _resolve_turn_from_ops equivalent -----------
        # Get opponent action (random from mask)
        opponent_mask = torch.FloatTensor(obs["player_1"]["action_mask"]).to(device)
        valid_actions = torch.where(opponent_mask > 0)[0]
        opp_action_id = int(np.random.choice(valid_actions.cpu().numpy())) if len(valid_actions) > 0 else 0
        del opponent_mask, valid_actions

        t2 = time.perf_counter()

        obs, rewards, terminated, truncated, info = env._resolve_turn_mixed(
            player_0_action_id=action_id,
            player_1_action_id=opp_action_id,
        )

        t3 = time.perf_counter()
        env_times.append((t3 - t2) * 1000)  # ms

        step_times.append((t3 - t0) * 1000)  # ms

        done = terminated or truncated
        if done:
            obs, _ = env.reset()

        step += 1
        if step % 100 == 0:
            logger.info(f"  Step {step}/{args.steps}...")

    # --- Print results -------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("  PHASE                Mean    Median    P95     P99     Min     Max")
    logger.info("  " + "-" * 58)

    def print_row(name, data):
        if len(data) < 2:
            logger.info(f"  {name:20s}  insufficient data")
            return
        data_sorted = sorted(data)
        logger.info(
            f"  {name:20s}  "
            f"{statistics.mean(data):>6.2f}  {statistics.median(data):>6.2f}  "
            f"{data_sorted[int(len(data_sorted)*0.95)]:>6.2f}  "
            f"{data_sorted[int(len(data_sorted)*0.99)]:>6.2f}  "
            f"{min(data):>6.2f}  {max(data):>6.2f}  ms"
        )

    print_row("GPU inference", gpu_times)
    print_row("Env step", env_times)
    print_row("Total step", step_times)

    if len(gpu_times) > 1 and len(env_times) > 1:
        total_gpu = sum(gpu_times)
        total_env = sum(env_times)
        gpu_pct = total_gpu / (total_gpu + total_env) * 100
        env_pct = total_env / (total_gpu + total_env) * 100
        logger.info("  " + "-" * 58)
        logger.info(f"  GPU: {gpu_pct:.1f}%  |  Env: {env_pct:.1f}%  |  "
                     f"GPU/Env ratio: 1:{total_env/total_gpu:.2f}")
        logger.info("=" * 65)

    # --- Simulate multi-env scenario -----------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("  MULTI-ENV ESTIMATION (num_envs=N)")
    logger.info("  " + "-" * 58)

    mean_gpu = statistics.mean(gpu_times)
    mean_env = statistics.mean(env_times)
    logger.info(f"  Measured mean GPU time per step:         {mean_gpu:.3f} ms")
    logger.info(f"  Measured mean Env time per step:         {mean_env:.3f} ms")
    logger.info(f"  Measured mean Total time per step:       {statistics.mean(step_times):.3f} ms")

    logger.info("")
    for N in [2, 4, 6, 8]:
        # Sequential env stepping: N × env_time
        seq_step_time = mean_gpu + N * mean_env
        # Estimated throughput
        trajs_per_sec = N / (seq_step_time * 250 / 1000)  # assuming 250 steps per episode
        logger.info(f"  N={N}:  seq_step={seq_step_time:.2f}ms  |  "
                     f"throughput={trajs_per_sec:.1f} traj/s  |  "
                     f"GPU%={mean_gpu/seq_step_time*100:.0f}%")

    logger.info("=" * 65)


if __name__ == "__main__":
    main()
