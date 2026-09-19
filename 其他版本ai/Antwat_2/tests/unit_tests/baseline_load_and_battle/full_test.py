#!/usr/bin/env python3
"""完整对战测试 - 测试所有Agent与BasicTowerAI的对战"""
import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')

from battle_simulator import BattleSimulator, BasicTowerAI, LOG_LEVEL_INFO, load_rule_based_agent
from model_registry import ModelRegistry
from model_loader import ModelLoader

print("=" * 70)
print("完整对战测试 - 所有Agent vs BasicTowerAI")
print("=" * 70)

registry = ModelRegistry()
registry.load_default_models()

loader = ModelLoader()
loaded_models = {}

print("\n" + "=" * 50)
print("模型加载阶段")
print("=" * 50)

for model_name in registry.list_models():
    config = registry.get_model_config(model_name)
    model_type = config.get('type', 'neural_network')

    if model_type == 'rule_based':
        agent = load_rule_based_agent(config)
        if agent is not None:
            loaded_models[model_name] = {'agent': agent, 'type': 'rule_based'}
            print(f"✓ 加载规则型AI: {model_name}")
    else:
        agent, hidden_dim = loader.load_model(config)
        if agent is not None:
            loaded_models[model_name] = {'agent': agent, 'hidden_dim': hidden_dim, 'type': 'neural_network'}
            print(f"✓ 加载神经网络模型: {model_name} (hidden_dim={hidden_dim})")

print(f"\n成功加载 {len(loaded_models)} 个模型: {list(loaded_models.keys())}")

simulator = BattleSimulator(log_level=LOG_LEVEL_INFO)
basic_tower = BasicTowerAI()

print("\n" + "=" * 50)
print("对战测试阶段")
print("=" * 50)

all_results = {}

for model_name, info in loaded_models.items():
    agent_type = info['type']
    agent = info['agent']

    print(f"\n--- 测试 {model_name} ---")

    result = simulator.battle(
        agent=agent,
        agent_name=model_name,
        agent_type=agent_type,
        opponent=basic_tower,
        opponent_name="BasicTowerAI",
        episodes=2,
        max_rounds=512,
        verbose=False
    )
    all_results[model_name] = result

print("\n" + "=" * 70)
print("最终汇总")
print("=" * 70)
print(f"{'模型':<15} {'胜':<6} {'负':<6} {'平':<6} {'胜率':<10} {'平均回合':<10}")
print("-" * 70)

for model_name, result in all_results.items():
    wins = result['agent_wins']
    losses = result['basic_wins']
    draws = result['draws']
    total = wins + losses + draws
    win_rate = wins / total * 100 if total > 0 else 0

    round_counts = [bd.get('total_rounds', 0) for bd in result['battle_details'] if bd.get('total_rounds', 0) > 0]
    avg_rounds = sum(round_counts) / len(round_counts) if round_counts else 0

    print(f"{model_name:<15} {wins:<6} {losses:<6} {draws:<6} {win_rate:>6.1f}%    {avg_rounds:>6.1f}")

print("=" * 70)
print("测试完成！")