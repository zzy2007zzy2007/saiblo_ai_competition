#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/ppo_v1/ppo/src')
sys.path.insert(0, '/root/autodl-tmp/AntWar/ppo_v1/Ant-Game')

from ppo_antwar.env.antwar_env import AntWarEnv
import numpy as np

def test_with_noop_opponent():
    """Test with NO_OP for opponent (like training when no opponent)"""
    print("=== Testing with NO_OP opponent (like training) ===")

    for episode in range(3):
        env = AntWarEnv()  # seed=None, resets to same state
        obs, _ = env.reset()
        print(f"\n--- Episode {episode} ---")

        ep_rewards = []
        for step in range(20):
            # Player 0 takes random actions (like policy)
            action = np.random.randint(0, 96)
            # Player 1 ALWAYS does NO_OP (like training when opponent is None)
            opponent_action = 0

            actions_dict = {"player_0": action, "player_1": opponent_action}
            obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            reward = rewards_dict.get("player_0", 0.0)
            ep_rewards.append(reward)
            if terminated or truncated:
                break

        print(f"Episode {episode}: sum={sum(ep_rewards):.4f}, mean={np.mean(ep_rewards):.4f}, max={max(ep_rewards):.4f}")

def test_with_policy_like_actions():
    """Test with actions that have high probability from a trained policy"""
    print("\n=== Testing with deterministic actions (like trained policy) ===")

    for episode in range(3):
        env = AntWarEnv()
        obs, _ = env.reset()
        print(f"\n--- Episode {episode} ---")

        ep_rewards = []
        for step in range(20):
            # Use same action every time (like a deterministic policy that always picks action 0)
            action = 0  # NO_OP for player 0 too!
            opponent_action = 0

            actions_dict = {"player_0": action, "player_1": opponent_action}
            obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
            reward = rewards_dict.get("player_0", 0.0)
            ep_rewards.append(reward)
            if terminated or truncated:
                break

        print(f"Episode {episode}: sum={sum(ep_rewards):.4f}, mean={np.mean(ep_rewards):.4f}, max={max(ep_rewards):.4f}")

def test_action_0_detail():
    """Test what action 0 actually does"""
    print("\n=== Testing what action 0 (NO_OP) does in detail ===")

    env = AntWarEnv()
    obs, _ = env.reset()

    # Check action mask
    player_obs = obs["player_0"]
    mask = player_obs["action_mask"]
    print(f"Action 0 valid: {mask[0] > 0}")
    print(f"Action 0-9 valid: {mask[:10]}")

    # Try action 0
    action = 0
    opponent_action = 0
    actions_dict = {"player_0": action, "player_1": opponent_action}
    obs, rewards_dict, terminated, truncated, info = env.step(actions_dict)
    print(f"After action 0: reward={rewards_dict.get('player_0', 0.0):.4f}, done={terminated or truncated}")

if __name__ == "__main__":
    test_with_noop_opponent()
    test_with_policy_like_actions()
    test_action_0_detail()
