"""详细对战测试 - 查看HP变化情况"""
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'baselines/ann_593'))

from SDK.backend.core import load_backend
from tests.evaluation.simple_action_catalog import SimpleActionCatalog
from tests.evaluation.baseline_strategy_loader import load_baseline_strategy

def main():
    print("=" * 80)
    print("详细对战测试 - 查看HP变化")
    print("=" * 80)
    print()
    
    # 加载策略
    agent1 = load_baseline_strategy('sample')
    agent2 = load_baseline_strategy('gen99')
    
    # 加载官方后端
    backend = load_backend(prefer_native=False)
    catalog = SimpleActionCatalog()
    
    # 创建初始状态
    state = backend.initial_state(seed=0)
    
    print(f"初始状态:")
    print(f"  sample HP: {state.bases[0].hp}")
    print(f"  gen99 HP: {state.bases[1].hp}")
    print(f"  sample 金币: {state.coins[0]}")
    print(f"  gen99 金币: {state.coins[1]}")
    print()
    
    # 运行对战，每10回合打印一次详细信息
    for round_num in range(1, 201):
        # 选择动作
        bundles0 = catalog.build(state, 0)
        bundle0 = agent1.choose_bundle(state, 0, bundles0)
        ops0 = list(bundle0.operations)
        
        bundles1 = catalog.build(state, 1)
        bundle1 = agent2.choose_bundle(state, 1, bundles1)
        ops1 = list(bundle1.operations)
        
        # 执行回合
        state.resolve_turn(ops0, ops1)
        
        # 每10回合打印一次
        if round_num % 10 == 0:
            print(f"回合 {round_num:3d}: sample HP={state.bases[0].hp:2d}, gen99 HP={state.bases[1].hp:2d}, "
                  f"金币: sample={state.coins[0]:3d}, gen99={state.coins[1]:3d}, "
                  f"蚂蚁: sample={len(state.ants_of(0)):2d}, gen99={len(state.ants_of(1)):2d}, "
                  f"防御塔: sample={state.tower_count(0):2d}, gen99={state.tower_count(1):2d}")
        
        # 检查是否结束
        if state.terminal:
            print(f"\n游戏结束！回合 {round_num}")
            print(f"胜利者: {'sample' if state.winner == 0 else 'gen99' if state.winner == 1 else '平局'}")
            break
    
    if not state.terminal:
        print(f"\n达到最大回合数 200，游戏未结束")
        print(f"最终HP: sample={state.bases[0].hp}, gen99={state.bases[1].hp}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
