"""对比测试：SimpleActionCatalog vs ActionCatalog"""
import sys
import os
import time

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'baselines/ann_593'))

from SDK.backend.core import load_backend
from SDK.utils.actions import ActionCatalog
from tests.evaluation.simple_action_catalog import SimpleActionCatalog
from tests.evaluation.baseline_strategy_loader import load_baseline_strategy

def test_catalog_performance(catalog_class, catalog_name, agent1, agent2, max_rounds=100):
    """测试特定ActionCatalog的性能"""
    backend = load_backend(prefer_native=False)
    catalog = catalog_class()
    
    state = backend.initial_state(seed=0)
    
    start_time = time.time()
    round_count = 0
    
    while not state.terminal and round_count < max_rounds:
        round_count += 1
        
        # 玩家0选择动作
        bundles0 = catalog.build(state, 0)
        bundle0 = agent1.choose_bundle(state, 0, bundles0)
        ops0 = list(bundle0.operations)
        
        # 玩家1选择动作
        bundles1 = catalog.build(state, 1)
        bundle1 = agent2.choose_bundle(state, 1, bundles1)
        ops1 = list(bundle1.operations)
        
        # 执行回合
        state.resolve_turn(ops0, ops1)
    
    elapsed_time = time.time() - start_time
    
    return {
        "catalog_name": catalog_name,
        "rounds": round_count,
        "time": elapsed_time,
        "time_per_round": elapsed_time / round_count if round_count > 0 else 0,
        "final_hp": (state.bases[0].hp, state.bases[1].hp),
        "winner": state.winner
    }

def main():
    print("=" * 80)
    print("对比测试：SimpleActionCatalog vs ActionCatalog")
    print("=" * 80)
    print()
    
    # 加载策略
    agent1 = load_baseline_strategy('sample')
    agent2 = load_baseline_strategy('gen99')
    
    print("测试配置:")
    print("  策略: sample vs gen99")
    print("  最大回合数: 100")
    print("  测试轮数: 3")
    print()
    
    # 测试SimpleActionCatalog
    print("=" * 80)
    print("测试 1: SimpleActionCatalog（禁用一步前瞻搜索）")
    print("=" * 80)
    
    simple_results = []
    for i in range(3):
        print(f"\n第 {i+1} 轮测试...")
        result = test_catalog_performance(SimpleActionCatalog, "SimpleActionCatalog", agent1, agent2, max_rounds=100)
        simple_results.append(result)
        print(f"  完成回合: {result['rounds']}")
        print(f"  耗时: {result['time']:.2f}秒")
        print(f"  每回合耗时: {result['time_per_round']:.4f}秒")
        print(f"  最终HP: sample={result['final_hp'][0]}, gen99={result['final_hp'][1]}")
    
    # 测试ActionCatalog
    print("\n" + "=" * 80)
    print("测试 2: ActionCatalog（启用一步前瞻搜索）")
    print("=" * 80)
    print("注意：此测试可能需要较长时间...")
    
    original_results = []
    for i in range(3):
        print(f"\n第 {i+1} 轮测试...")
        try:
            result = test_catalog_performance(ActionCatalog, "ActionCatalog", agent1, agent2, max_rounds=100)
            original_results.append(result)
            print(f"  完成回合: {result['rounds']}")
            print(f"  耗时: {result['time']:.2f}秒")
            print(f"  每回合耗时: {result['time_per_round']:.4f}秒")
            print(f"  最终HP: sample={result['final_hp'][0]}, gen99={result['final_hp'][1]}")
        except KeyboardInterrupt:
            print("  测试被中断（耗时过长）")
            break
    
    # 对比结果
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    
    if simple_results:
        avg_time_simple = sum(r['time'] for r in simple_results) / len(simple_results)
        avg_time_per_round_simple = sum(r['time_per_round'] for r in simple_results) / len(simple_results)
        print(f"\nSimpleActionCatalog:")
        print(f"  平均总耗时: {avg_time_simple:.2f}秒")
        print(f"  平均每回合耗时: {avg_time_per_round_simple:.4f}秒")
    
    if original_results:
        avg_time_original = sum(r['time'] for r in original_results) / len(original_results)
        avg_time_per_round_original = sum(r['time_per_round'] for r in original_results) / len(original_results)
        print(f"\nActionCatalog:")
        print(f"  平均总耗时: {avg_time_original:.2f}秒")
        print(f"  平均每回合耗时: {avg_time_per_round_original:.4f}秒")
        
        if simple_results:
            speedup = avg_time_original / avg_time_simple
            print(f"\n性能提升: {speedup:.2f}x 倍")
    
    # 分析影响
    print("\n" + "=" * 80)
    print("影响分析")
    print("=" * 80)
    
    print("\n一步前瞻搜索的作用:")
    print("  1. 对每个候选动作模拟一回合")
    print("  2. 使用特征提取器评估状态变化")
    print("  3. 根据评估结果调整动作分数")
    print("  4. 提供更精准的动作选择")
    
    print("\n禁用一步前瞻搜索的影响:")
    print("  优点:")
    print("    - 显著提高性能（预计10-100倍加速）")
    print("    - 降低计算资源消耗")
    print("    - 适合大规模测试和训练")
    print("  缺点:")
    print("    - 动作选择可能不够精准")
    print("    - 策略表现可能略有下降")
    print("    - 对复杂局势的判断能力降低")
    
    print("\n建议:")
    print("  - 训练阶段：使用SimpleActionCatalog（提高效率）")
    print("  - 最终评估：使用ActionCatalog（确保准确性）")
    print("  - 快速测试：使用SimpleActionCatalog（节省时间）")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
