import torch
import numpy as np
import os
import sys
import time

# 添加 saiblo-antwar-sdk-python 目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../saiblo-antwar-sdk-python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tests/strategies'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../ppo_v2'))

from ppo_v2.environment import AntWarEnv
from ppo_v2.model import PPOAgent
from basic_tower import BasicTowerAI
from ann_v1 import AnnV1
from ann_593 import Ann593
from gen99 import Gen99
from gen199 import Gen199

def test_against_baseline(model_path, baseline_agent, episodes=5):
    """测试模型与指定baseline策略的对战"""
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 环境
    env = AntWarEnv()
    action_dim = env.get_action_space()
    
    # 加载模型
    agent = PPOAgent(action_dim, device)
    if os.path.exists(model_path):
        agent.load(model_path)
    else:
        print(f"Model file {model_path} not found!")
        return None
    
    # 测试结果
    results = {
        'total_episodes': episodes,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'total_reward': 0,
        'avg_reward': 0,
        'avg_rounds': 0
    }
    
    start_time = time.time()
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        done = False
        round_count = 0
        final_reward = 0
        
        while not done:
            round_count += 1
            
            # 选择动作（使用确定性策略）
            action, _, _ = agent.select_action(state, deterministic=True)
            
            # 敌方选择动作（baseline策略）
            enemy_actions = baseline_agent.select_action(env.game_state, 1)
            
            # 执行动作
            next_state, reward, done = env.step(action, enemy_actions, episode=1000)  # 使用稀疏奖励机制
            
            # 更新状态
            state = next_state
            
            # 只记录游戏结束时的最终奖励
            if done:
                final_reward = reward
        
        # 记录结果
        results['total_reward'] += final_reward
        results['avg_rounds'] += round_count
        
        # 胜负判断（基于基地血量）
        if env.game_state.hp[0] > 0 and env.game_state.hp[1] <= 0:
            # 我方获胜
            results['wins'] += 1
        elif env.game_state.hp[0] <= 0 and env.game_state.hp[1] > 0:
            # 我方失败
            results['losses'] += 1
        else:
            # 平局
            results['draws'] += 1
    
    # 计算平均值
    results['avg_reward'] = results['total_reward'] / episodes
    results['avg_rounds'] = results['avg_rounds'] / episodes
    results['win_rate'] = results['wins'] / episodes * 100
    results['test_time'] = time.time() - start_time
    
    return results

def main():
    """测试模型与所有baseline策略的对战"""
    # 模型路径
    model_path = "models/ppo_best_model.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        # 尝试从上级目录读取
        model_path = "../ppo_best_model.pth"
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found!")
            return
    
    # 测试轮数
    episodes = 5
    
    # 基线策略列表
    baselines = [
        BasicTowerAI(),
        AnnV1(),
        Ann593(),
        Gen99(),
        Gen199()
    ]
    
    # 测试结果汇总
    all_results = {}
    
    print("=== Testing model against all baseline strategies ===")
    print(f"Testing for {episodes} episodes per baseline...")
    print()
    
    for baseline in baselines:
        print(f"Testing against {baseline.name}...")
        results = test_against_baseline(model_path, baseline, episodes)
        if results:
            all_results[baseline.name] = results
            print(f"  Wins: {results['wins']} ({results['win_rate']:.2f}%)")
            print(f"  Losses: {results['losses']}")
            print(f"  Draws: {results['draws']}")
            print(f"  Avg Reward: {results['avg_reward']:.2f}")
            print(f"  Avg Rounds: {results['avg_rounds']:.2f}")
            print(f"  Test Time: {results['test_time']:.2f}s")
            print()
    
    # 打印汇总结果
    print("=== Summary Results ===")
    print("Baseline Strategy | Win Rate | Avg Reward | Avg Rounds")
    print("-" * 60)
    for name, result in all_results.items():
        print(f"{name:<16} | {result['win_rate']:>8.2f}% | {result['avg_reward']:>10.2f} | {result['avg_rounds']:>10.2f}")
    
    # 计算整体胜率
    total_wins = sum(r['wins'] for r in all_results.values())
    total_episodes = sum(r['total_episodes'] for r in all_results.values())
    overall_win_rate = total_wins / total_episodes * 100
    
    print("-" * 60)
    print(f"Overall Win Rate: {overall_win_rate:.2f}%")

if __name__ == "__main__":
    main()