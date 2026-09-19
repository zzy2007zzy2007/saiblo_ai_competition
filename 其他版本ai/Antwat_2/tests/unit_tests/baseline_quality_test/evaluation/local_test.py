"""本地小规模测试 - 只测试不使用神经网络的策略"""
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
    print("本地小规模测试 - 不使用神经网络的策略")
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
    
    # 运行一个简单的对战测试
    print("=== 开始对战测试 ===")
    print("测试: sample vs gen99")
    print("对战轮数: 2")
    print("最大回合数: 50")
    print()
    
    try:
        wins1, wins2, draws, avg_rounds = OfficialBattleEngine.run_battle(
            agents['sample'],
            agents['gen99'],
            episodes=2,
            max_rounds=50,
            verbose=True,
            log_file=None
        )
        
        print()
        print("=" * 80)
        print("对战测试完成")
        print("=" * 80)
        print(f"sample 胜场: {wins1}")
        print(f"gen99 胜场: {wins2}")
        print(f"平局: {draws}")
        print(f"平均回合数: {avg_rounds:.2f}")
        
    except Exception as e:
        print(f"对战失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
