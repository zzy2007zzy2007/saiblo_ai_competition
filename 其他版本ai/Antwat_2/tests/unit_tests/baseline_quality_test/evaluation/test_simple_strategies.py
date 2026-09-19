"""简单策略测试脚本 - 只测试不需要神经网络的策略"""
import argparse
import json
import sys
import os
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from tests.evaluation.official_battle_engine import OfficialBattleEngine
from tests.evaluation.baseline_strategy_loader import load_baseline_strategy

def main():
    print("=" * 80)
    print("简单策略对战测试")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 只加载简单策略
    simple_strategies = ['sample', 'gen99', 'gen199']
    
    print("=== 加载策略 ===")
    agents = {}
    for strategy_name in simple_strategies:
        try:
            agent = load_baseline_strategy(strategy_name)
            agents[strategy_name] = agent
            print(f"✓ 成功加载策略: {strategy_name}")
        except Exception as e:
            print(f"✗ 加载策略 {strategy_name} 失败: {e}")
    
    if len(agents) < 2:
        print("错误: 至少需要2个策略才能进行比较")
        return
    
    print(f"\n成功加载 {len(agents)} 个策略")
    print()
    
    # 创建对战日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "battle_logs_official")
    os.makedirs(log_dir, exist_ok=True)
    
    # 运行对战
    print("=== 开始对战 ===")
    results = []
    
    strategy_names = list(agents.keys())
    total_pairs = len(strategy_names) * (len(strategy_names) - 1)
    current_pair = 0
    
    for i, strategy1 in enumerate(strategy_names):
        for j, strategy2 in enumerate(strategy_names):
            if i == j:
                continue
            
            current_pair += 1
            print(f"\n[{current_pair}/{total_pairs}] {strategy1} vs {strategy2}")
            
            # 创建日志文件
            log_file = os.path.join(log_dir, f"{strategy1}_vs_{strategy2}.log")
            
            # 运行对战（减少参数）
            try:
                wins1, wins2, draws, avg_rounds = OfficialBattleEngine.run_battle(
                    agents[strategy1],
                    agents[strategy2],
                    episodes=5,  # 减少到5轮
                    max_rounds=100,  # 减少到100回合
                    verbose=True,
                    log_file=log_file
                )
                
                # 记录结果
                result = {
                    "agent1": strategy1,
                    "agent2": strategy2,
                    "wins1": wins1,
                    "wins2": wins2,
                    "draws": draws,
                    "win_rate1": wins1 / 5 * 100,
                    "win_rate2": wins2 / 5 * 100,
                    "draw_rate": draws / 5 * 100,
                    "avg_rounds": avg_rounds,
                    "episodes": 5,
                    "log_file": f"battle_logs_official/{strategy1}_vs_{strategy2}.log"
                }
                results.append(result)
                
                print(f"  结果: {strategy1} {wins1}胜, {strategy2} {wins2}胜, {draws}平")
                
            except Exception as e:
                print(f"  对战失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 导出结果
    export_path = os.path.join(os.path.dirname(__file__), "win_rates_simple.json")
    with open(export_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已导出到: {export_path}")
    
    print("\n" + "=" * 80)
    print("对战测试完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
