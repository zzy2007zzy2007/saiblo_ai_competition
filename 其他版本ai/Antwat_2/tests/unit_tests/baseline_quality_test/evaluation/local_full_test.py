"""本地完整测试 - 测试所有不使用神经网络的策略对"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'baselines/ann_593'))

from tests.evaluation.official_battle_engine import OfficialBattleEngine
from tests.evaluation.baseline_strategy_loader import load_baseline_strategy

def main():
    print("=" * 80)
    print("本地完整测试 - 不使用神经网络的策略")
    print("=" * 80)
    print()
    
    # 只加载不使用神经网络的策略
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
            import traceback
            traceback.print_exc()
    
    if len(agents) < 2:
        print("错误: 至少需要2个策略才能进行比较")
        return
    
    print(f"\n成功加载 {len(agents)} 个策略")
    print()
    
    # 运行所有策略对的对战
    print("=== 开始对战测试 ===")
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
            
            try:
                wins1, wins2, draws, avg_rounds = OfficialBattleEngine.run_battle(
                    agents[strategy1],
                    agents[strategy2],
                    episodes=3,  # 3轮对战
                    max_rounds=100,  # 100回合
                    verbose=False,
                    log_file=None
                )
                
                result = {
                    "agent1": strategy1,
                    "agent2": strategy2,
                    "wins1": wins1,
                    "wins2": wins2,
                    "draws": draws,
                    "win_rate1": wins1 / 3 * 100,
                    "win_rate2": wins2 / 3 * 100,
                    "avg_rounds": avg_rounds
                }
                results.append(result)
                
                print(f"  结果: {strategy1} {wins1}胜, {strategy2} {wins2}胜, {draws}平")
                print(f"  平均回合数: {avg_rounds:.2f}")
                
            except Exception as e:
                print(f"  对战失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 打印总结
    print("\n" + "=" * 80)
    print("对战测试完成 - 结果总结")
    print("=" * 80)
    
    # 统计每个策略的总胜场
    total_wins = {name: 0 for name in strategy_names}
    total_games = {name: 0 for name in strategy_names}
    
    for result in results:
        total_wins[result["agent1"]] += result["wins1"]
        total_games[result["agent1"]] += 3
        total_wins[result["agent2"]] += result["wins2"]
        total_games[result["agent2"]] += 3
    
    # 计算胜率
    win_rates = {name: total_wins[name] / total_games[name] * 100 if total_games[name] > 0 else 0 
                 for name in strategy_names}
    
    # 排序
    sorted_strategies = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
    
    print("\n策略性能排名:")
    print("-" * 80)
    for rank, (name, win_rate) in enumerate(sorted_strategies, 1):
        print(f"{rank}. {name:<12} | 胜率: {win_rate:5.2f}% | 胜场: {total_wins[name]}")
    
    print("\n头对头对战结果:")
    print("-" * 80)
    for result in results:
        print(f"\n{result['agent1']:<12} vs {result['agent2']:<12}")
        print(f"  {result['agent1']}: {result['win_rate1']:6.2f}% ({result['wins1']}胜)")
        print(f"  {result['agent2']}: {result['win_rate2']:6.2f}% ({result['wins2']}胜)")
        print(f"  平局: {result['draws']}场")
        print(f"  平均回合数: {result['avg_rounds']:.2f}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
