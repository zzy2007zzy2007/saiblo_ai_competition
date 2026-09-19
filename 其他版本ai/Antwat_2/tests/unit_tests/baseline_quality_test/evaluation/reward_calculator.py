#!/usr/bin/env python3
"""
reward_calculator.py - 基于胜率的公平奖励计算

该脚本用于基于策略之间的胜率数据，计算公平的正分和负分奖励值。
"""

import os
import sys
import json
import math
import numpy as np

class RewardCalculator:
    """基于胜率的奖励计算器"""
    
    def __init__(self, win_rates_file):
        """初始化奖励计算器
        
        Args:
            win_rates_file: 胜率数据文件路径
        """
        self.win_rates_file = win_rates_file
        self.win_rates = self.load_win_rates()
        self.strength_scores = self.calculate_strength_scores()
    
    def load_win_rates(self):
        """加载胜率数据"""
        if not os.path.exists(self.win_rates_file):
            raise FileNotFoundError(f"胜率数据文件不存在: {self.win_rates_file}")
        
        with open(self.win_rates_file, "r") as f:
            return json.load(f)
    
    def calculate_strength_scores(self):
        """计算每个策略的相对强度得分"""
        # 计算每个策略的平均胜率
        avg_win_rates = {}
        for agent_name, rates in self.win_rates.items():
            if rates:
                avg_win_rates[agent_name] = sum(rates.values()) / len(rates.values())
            else:
                avg_win_rates[agent_name] = 0
        
        # 归一化强度得分到[0,1]
        if not avg_win_rates:
            return {}
        
        min_rate = min(avg_win_rates.values())
        max_rate = max(avg_win_rates.values())
        
        strength_scores = {}
        if max_rate > min_rate:
            for agent_name, rate in avg_win_rates.items():
                strength_scores[agent_name] = (rate - min_rate) / (max_rate - min_rate)
        else:
            # 所有策略强度相同
            for agent_name in avg_win_rates:
                strength_scores[agent_name] = 0.5
        
        return strength_scores
    
    def calculate_positive_reward(self, opponent_name, base_reward=1000):
        """计算击败对手的正分奖励
        
        Args:
            opponent_name: 对手策略名称
            base_reward: 基础奖励值
            
        Returns:
            float: 正分奖励值
        """
        if opponent_name not in self.strength_scores:
            return base_reward
        
        # 强度得分，值越大表示对手越强
        strength = self.strength_scores[opponent_name]
        
        # 使用对数缩放函数，确保击败强对手获得更高奖励
        epsilon = 1e-6
        reward = base_reward * math.log(1 + strength / epsilon)
        
        # 确保奖励值在合理范围内
        min_reward = base_reward
        max_reward = base_reward * 5  # 最强对手的奖励是基础奖励的5倍
        
        return max(min_reward, min(max_reward, reward))
    
    def calculate_negative_reward(self, opponent_name, base_penalty=1000):
        """计算输给对手的负分惩罚
        
        Args:
            opponent_name: 对手策略名称
            base_penalty: 基础惩罚值
            
        Returns:
            float: 负分惩罚值（负数）
        """
        if opponent_name not in self.strength_scores:
            return -base_penalty
        
        # 强度得分，值越大表示对手越强
        strength = self.strength_scores[opponent_name]
        
        # 使用对数缩放函数，确保输给弱对手获得更大惩罚
        epsilon = 1e-6
        # 输给强对手的惩罚较小，输给弱对手的惩罚较大
        penalty = base_penalty * math.log(1 + (1 - strength) / epsilon)
        
        # 确保惩罚值在合理范围内
        min_penalty = base_penalty * 0.2  # 输给最强对手的惩罚是基础惩罚的20%
        max_penalty = base_penalty  # 输给最弱对手的惩罚是基础惩罚
        
        return -max(min_penalty, min(max_penalty, penalty))
    
    def calculate_reward(self, opponent_name, win, base_reward=1000, base_penalty=1000):
        """计算奖励值
        
        Args:
            opponent_name: 对手策略名称
            win: 是否获胜
            base_reward: 基础奖励值
            base_penalty: 基础惩罚值
            
        Returns:
            float: 奖励值（正数表示奖励，负数表示惩罚）
        """
        if win:
            return self.calculate_positive_reward(opponent_name, base_reward)
        else:
            return self.calculate_negative_reward(opponent_name, base_penalty)
    
    def print_strength_scores(self):
        """打印策略强度得分"""
        print("=== 策略强度得分 ===")
        sorted_strategies = sorted(
            self.strength_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for i, (strategy, score) in enumerate(sorted_strategies):
            print(f"{i+1}. {strategy:12} | 强度得分: {score:.4f}")
    
    def print_reward_table(self, base_reward=1000, base_penalty=1000):
        """打印奖励表"""
        print("\n=== 奖励表 ===")
        print(f"基础奖励: {base_reward}, 基础惩罚: {base_penalty}")
        print("-" * 60)
        print(f"{'策略':<12} | {'强度得分':<10} | {'击败奖励':<10} | {'输给惩罚':<10}")
        print("-" * 60)
        
        for strategy, score in self.strength_scores.items():
            positive_reward = self.calculate_positive_reward(strategy, base_reward)
            negative_reward = self.calculate_negative_reward(strategy, base_penalty)
            print(f"{strategy:<12} | {score:.4f}      | {positive_reward:.2f}      | {negative_reward:.2f}")
        
        print("-" * 60)

def generate_win_rates_file():
    """生成胜率数据文件"""
    # 使用现有的compare_strategies.py脚本生成胜率数据
    import subprocess
    import tempfile
    
    # 创建临时配置文件
    config = {
        "strategies": ["basic_tower", "ann_v1", "ann_593", "gen99", "gen199"],
        "episodes": 100,
        "max_rounds": 500,
        "export": "win_rates.json",
        "verbose": False
    }
    
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    import json
    json.dump(config, config_file)
    config_file.close()
    
    # 运行compare_strategies.py
    script_path = os.path.join(os.path.dirname(__file__), "compare_strategies.py")
    result = subprocess.run(
        [sys.executable, script_path, "--config", config_file.name],
        capture_output=True,
        text=True
    )
    
    # 清理临时文件
    os.unlink(config_file.name)
    
    if result.returncode == 0:
        print("胜率数据生成成功")
        return "win_rates.json"
    else:
        print("胜率数据生成失败:")
        print(result.stderr)
        return None

def main():
    """主函数"""
    # 检查是否存在胜率数据文件
    win_rates_file = os.path.join(os.path.dirname(__file__), "win_rates.json")
    
    if not os.path.exists(win_rates_file):
        print("胜率数据文件不存在，正在生成...")
        win_rates_file = generate_win_rates_file()
        if not win_rates_file:
            print("无法生成胜率数据，退出程序")
            return
    
    # 创建奖励计算器
    calculator = RewardCalculator(win_rates_file)
    
    # 打印强度得分
    calculator.print_strength_scores()
    
    # 打印奖励表
    calculator.print_reward_table()
    
    # 示例计算
    print("\n=== 奖励计算示例 ===")
    test_opponents = ["basic_tower", "ann_v1", "ann_593", "gen99", "gen199"]
    
    for opponent in test_opponents:
        win_reward = calculator.calculate_reward(opponent, True)
        lose_reward = calculator.calculate_reward(opponent, False)
        print(f"击败 {opponent}: {win_reward:.2f}")
        print(f"输给 {opponent}: {lose_reward:.2f}")
        print()

if __name__ == "__main__":
    main()