#!/usr/bin/env python3
"""
测试入口 - 重构版

使用重构后的BattleSimulator，支持：
1. 正确的 resolve_turn() API
2. 正确的胜负判定
3. 正确的游戏结束判定
4. 详细的日志分级
5. 完整的battle_details返回
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_registry import ModelRegistry
from model_loader import ModelLoader
from battle_simulator import BattleSimulator, load_rule_based_agent, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO

def main():
    parser = argparse.ArgumentParser(description='Baseline模型加载和对战测试')
    parser.add_argument('--episodes', type=int, default=2, help='每模型对战局数')
    parser.add_argument('--max-rounds', type=int, default=512, help='最大回合数')
    parser.add_argument('--debug', action='store_true', help='启用DEBUG日志级别')
    parser.add_argument('--model', type=str, default=None, help='仅测试指定模型')
    parser.add_argument('--opponent', type=str, default='BasicTowerAI', help='对手类型')
    args = parser.parse_args()

    log_level = LOG_LEVEL_DEBUG if args.debug else LOG_LEVEL_INFO

    print("=" * 70)
    print("Baseline 模型加载和对战测试 (重构版)")
    print("=" * 70)
    print(f"日志级别: {'DEBUG' if args.debug else 'INFO'}")
    print(f"每模型对战局数: {args.episodes}")
    print(f"最大回合数: {args.max_rounds}")
    print()

    registry = ModelRegistry()
    registry.load_default_models()

    loader = ModelLoader()
    loaded_models = {}

    print("\n" + "=" * 50)
    print("模型加载阶段")
    print("=" * 50)

    for model_name in registry.list_models():
        if args.model and args.model != model_name:
            continue

        config = registry.get_model_config(model_name)
        model_type = config.get('type', 'neural_network')

        if model_type == 'rule_based':
            agent = load_rule_based_agent(config)
            if agent is not None:
                loaded_models[model_name] = {'agent': agent, 'type': 'rule_based'}
        else:
            agent, hidden_dim = loader.load_model(config)
            if agent is not None:
                loaded_models[model_name] = {'agent': agent, 'hidden_dim': hidden_dim, 'type': 'neural_network'}

    if not loaded_models:
        print("错误: 没有成功加载任何模型！")
        return

    print(f"\n成功加载 {len(loaded_models)} 个模型: {list(loaded_models.keys())}")

    print("\n" + "=" * 50)
    print("对战测试阶段")
    print("=" * 50)

    simulator = BattleSimulator(log_level=log_level)
    all_results = {}

    for model_name, info in loaded_models.items():
        agent_type = info['type']
        agent = info['agent']

        result = simulator.battle(
            agent=agent,
            agent_name=model_name,
            agent_type=agent_type,
            opponent=None,
            opponent_name=args.opponent,
            episodes=args.episodes,
            max_rounds=args.max_rounds,
            verbose=True
        )
        all_results[model_name] = result

    print("\n" + "=" * 50)
    print("最终汇总")
    print("=" * 50)

    for model_name, result in all_results.items():
        wins = result['agent_wins']
        losses = result['basic_wins']
        draws = result['draws']
        total = wins + losses + draws
        win_rate = wins / total * 100 if total > 0 else 0
        print(f"{model_name}: {wins}胜/{losses}负/{draws}平 (胜率: {win_rate:.1f}%)")

    print("\n测试完成！")

if __name__ == "__main__":
    main()