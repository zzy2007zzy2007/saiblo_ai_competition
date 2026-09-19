#!/usr/bin/env python
"""测试我们的修复是否有效"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ppo', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Ant-Game'))

from ppo_antwar.env.antwar_env import AntWarEnv

def test_env():
    print("初始化环境...")
    env = AntWarEnv()
    
    print("重置环境...")
    obs, info = env.reset()
    
    print("初始信息:", info)
    
    print("\n执行几个 step...")
    for i in range(5):
        actions = {'player_0': 0, 'player_1': 0}
        obs, rewards, terminated, truncated, info = env.step(actions)
        print(f"Step {i+1}:")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Info: {info}")
        if terminated or truncated:
            break
    
    print("\n测试完成！")

if __name__ == "__main__":
    test_env()
