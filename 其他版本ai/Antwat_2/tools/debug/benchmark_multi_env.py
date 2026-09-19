#!/usr/bin/env python3
"""
Benchmark: measure actual end-to-end throughput of multi-environment SelfPlay.

Tests multiple configurations to evaluate how much throughput gain
multi-env parallelism can provide:
  1. Baseline:    1 env, sequential (current approach)
  2. SeqStep:     N envs, sequential stepping, NO batch inference
  3. BatchSeq:    N envs, sequential stepping, WITH batch inference
  4. BatchThread: N envs, threaded stepping, WITH batch inference

Usage:
    python tools/debug/benchmark_multi_env.py [--steps 500] [--device cuda] [--n-envs 1 2 4 6]

Output:
    Wall-clock throughput (traj/s) for each configuration.
"""

import argparse
import sys
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import torch

# --- Setup PYTHONPATH -------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PPO_SRC = REPO_ROOT / "ppo" / "src"
if str(PPO_SRC) not in sys.path:
    sys.path.insert(0, str(PPO_SRC))

# --- SDK path ----------------------------------------------------------
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


def create_policy(device):
    from ppo_antwar.utils import get_config
    config = get_config()
    from ppo_antwar.network.antwar_net import AntWarPolicyValueNetwork
    policy = AntWarPolicyValueNetwork(
        board_shape=tuple(config.network.board_shape),
        global_dim=config.network.global_dim,
        action_dim=config.network.action_dim,
        hidden_dim=config.network.hidden_dim,
    ).to(device)
    policy.eval()
    return policy


def select_action(policy, obs, device):
    """Single env action selection (current approach)"""
    board = torch.FloatTensor(obs['board']).unsqueeze(0).to(device)
    global_obs = torch.FloatTensor(obs['global']).unsqueeze(0).to(device)
    mask = torch.FloatTensor(obs['action_mask']).unsqueeze(0).to(device)

    with torch.no_grad():
        action, log_prob, value = policy.get_action(
            board, global_obs, mask, deterministic=False
        )
    return action.cpu().item(), log_prob.cpu().item(), value.cpu().item()


def select_actions_batch(policy, obs_list, device):
    """Batch action selection for multiple envs"""
    boards = torch.FloatTensor(np.array([o['board'] for o in obs_list])).to(device)
    globals = torch.FloatTensor(np.array([o['global'] for o in obs_list])).to(device)
    masks = torch.FloatTensor(np.array([o['action_mask'] for o in obs_list])).to(device)

    with torch.no_grad():
        # Use get_action but with batched input
        # get_action expects (N, C, H, W) already
        actions, log_probs, values = policy.get_action(
            boards, globals, masks, deterministic=False
        )
    return (
        actions.cpu().tolist(),
        log_probs.cpu().tolist(),
        values.cpu().tolist(),
    )


def get_random_opponent_action(obs, device):
    """Random action from opponent mask"""
    mask = torch.FloatTensor(obs['action_mask']).to(device)
    valid = torch.where(mask > 0)[0]
    if len(valid) > 0:
        return int(np.random.choice(valid.cpu().numpy()))
    return 0


def step_env(env, action_id, opp_action_id, swap):
    """Step a single env with given actions"""
    if not swap:
        obs, rewards, terminated, truncated, info = env._resolve_turn_mixed(
            player_0_action_id=action_id,
            player_1_action_id=opp_action_id,
        )
    else:
        obs, rewards, terminated, truncated, info = env._resolve_turn_mixed(
            player_0_action_id=opp_action_id,
            player_1_action_id=action_id,
        )
    return obs, rewards, terminated, truncated, info


def benchmark_baseline(policy, device, num_steps, max_steps_per_ep=512):
    """Configuration 1: 1 env, sequential (current approach)"""
    from ppo_antwar.env.antwar_env import AntWarEnv
    env = AntWarEnv()

    obs, _ = env.reset()
    steps_done = 0
    total_time = 0.0

    while steps_done < num_steps:
        player_obs = obs["player_0"]

        t0 = time.perf_counter()
        action_id, _, _ = select_action(policy, player_obs, device)
        opp_action_id = get_random_opponent_action(obs["player_1"], device)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        obs, rewards, terminated, truncated, info = step_env(env, action_id, opp_action_id, False)
        t2 = time.perf_counter()

        total_time += (t2 - t0)
        steps_done += 1

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return steps_done / total_time  # steps/s
    # Note: traj/s = steps/s / avg_steps_per_episode


def benchmark_batch_seq(policy, device, num_steps, n_envs, max_steps_per_ep=512):
    """Configuration 2: N envs, sequential stepping, WITH batch inference"""
    from ppo_antwar.env.antwar_env import AntWarEnv
    envs = [AntWarEnv() for _ in range(n_envs)]

    obs_list = [env.reset()[0] for env in envs]
    active = [True] * n_envs
    step_counts = [0] * n_envs
    total_steps = 0
    total_time = 0.0

    while total_steps < num_steps:
        # Collect active envs' obs
        active_idx = [i for i, a in enumerate(active) if a]
        if not active_idx:
            # All done, reset all
            for i in range(n_envs):
                obs_list[i], _ = envs[i].reset()
                active[i] = True
                step_counts[i] = 0
            continue

        t0 = time.perf_counter()

        # Batch inference
        batch_obs = [obs_list[i]["player_0"] for i in active_idx]
        actions, _, _ = select_actions_batch(policy, batch_obs, device)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Sequential stepping
        for j, i in enumerate(active_idx):
            opp_action = get_random_opponent_action(obs_list[i]["player_1"], device)
            obs_list[i], _, terminated, truncated, _ = step_env(
                envs[i], actions[j], opp_action, False
            )
            step_counts[i] += 1
            total_steps += 1

            if terminated or truncated or step_counts[i] >= max_steps_per_ep:
                obs_list[i], _ = envs[i].reset()
                # Keep active (new episode starts immediately)
                step_counts[i] = 0

        t2 = time.perf_counter()
        total_time += (t2 - t0)

        if total_steps >= num_steps:
            break

    for env in envs:
        env.close()

