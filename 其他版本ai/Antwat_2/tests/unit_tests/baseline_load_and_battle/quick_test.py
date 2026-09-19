#!/usr/bin/env python3
"""完整测试 - 512回合对战"""
import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')

from battle_simulator import BattleSimulator, BasicTowerAI, LOG_LEVEL_INFO

print("=" * 60)
print("完整测试 - 512回合对战 (BasicTowerAI vs BasicTowerAI)")
print("=" * 60)

simulator = BattleSimulator(log_level=LOG_LEVEL_INFO)
basic_tower = BasicTowerAI()

result = simulator.battle(
    agent=basic_tower,
    agent_name="BasicTowerAI",
    agent_type="rule_based",
    opponent=basic_tower,
    opponent_name="BasicTowerAI",
    episodes=1,
    max_rounds=512,
    verbose=True
)

print("\n对战结果:")
print(f"  胜: {result['agent_wins']}")
print(f"  负: {result['basic_wins']}")
print(f"  平: {result['draws']}")

if result['battle_details']:
    bd = result['battle_details'][0]
    print(f"\n详细:")
    print(f"  总回合数: {bd.get('total_rounds', 'N/A')}")
    print(f"  最终HP: agent={bd['final_hp']['agent']}, opponent={bd['final_hp']['opponent']}")
    print(f"  结果: {bd['result']}")
    print(f"  耗时: {bd['duration']:.2f}秒")