#!/usr/bin/env python3
"""修复问题并深入分析对战过程"""
import os
import sys
import time
import random
from datetime import datetime


BASE_DIR = '/root/autodl-tmp/AntWar'
MAIN_SDK_PATH = os.path.join(BASE_DIR, 'ppo_v1', 'Ant-Game', 'SDK')


def print_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")


def get_action_name(op_type):
    """获取动作类型名称"""
    action_names = {
        1: 'SPAWN_ANT',
        2: 'BUILD_TOWER',
        3: 'UPGRADE_TOWER',
        4: 'DEMOLISH_TOWER',
        5: 'UPGRADE_BASE',
        6: 'COLLECT_FOOD',
        7: 'COLLECT_GOLD',
        8: 'SET_FOCUS_POINT',
        9: 'SET_PATROL_PATH',
        10: 'USE_ABILITY',
        11: 'BUY_BUNDLE',
        12: 'SELL_TOWER',
        13: 'RESEARCH',
        14: 'RECRUIT_HERO',
        15: 'EQUIP_ITEM',
    }
    return action_names.get(op_type, f'UNKNOWN_{op_type}')


class BasicRandomAI:
    def __init__(self):
        self.sdk_path = MAIN_SDK_PATH
        sys.path.insert(0, self.sdk_path)
        from SDK.utils.actions import ActionCatalog
        self.ActionCatalog = ActionCatalog

    def choose_operations(self, state, player):
        try:
            catalog = self.ActionCatalog(max_actions=32)
            bundles = catalog.build(state, player)
            if bundles and len(bundles) > 0:
                selected = random.choice(bundles)
                ops = list(selected.operations)
                if ops:
                    op_type = ops[0].op_type.value if hasattr(ops[0].op_type, 'value') else ops[0].op_type
                    print_log(f"  RandomAI 金币={state.coins[player]} 选择动作: {get_action_name(op_type)} ({op_type})")
                return ops
        except Exception as e:
            print_log(f"BasicRandomAI 错误: {e}")
        return []


def load_baseline_model(model_name):
    model_path = os.path.join(BASE_DIR, 'baselines', model_name)
    sdk_path = os.path.join(model_path, 'SDK')

    original_path = sys.path.copy()
    try:
        sys.path = []
        sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)
        for p in original_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)

        from ai import AI
        agent = AI()
        print_log(f"✓ 加载 {model_name} AI 成功")
        sys.path = original_path
        return agent, model_path, sdk_path, original_path
    except Exception as e:
        print_log(f"✗ 加载失败: {e}")
        sys.path = original_path
        return None


def get_baseline_ops(agent, state, player, model_path, sdk_path, saved_path, model_name):
    """修复: 添加 model_name 参数"""
    temp_path = sys.path.copy()
    try:
        sys.path = []
        sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)
        for p in saved_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)

        if hasattr(agent, 'choose_bundle') and hasattr(agent, 'list_bundles'):
            bundles = agent.list_bundles(state, player)
            if bundles:
                bundle = agent.choose_bundle(state, player, bundles)
                ops = list(bundle.operations)
                if ops:
                    op_type = ops[0].op_type.value if hasattr(ops[0].op_type, 'value') else ops[0].op_type
                    print_log(f"  {model_name} 金币={state.coins[player]} 选择动作: {get_action_name(op_type)} ({op_type})")
                return ops
            else:
                print_log(f"  {model_name} 没有可用动作")
        elif hasattr(agent, 'choose_operations'):
            ops = agent.choose_operations(state, player)
            if ops:
                op_type = ops[0].op_type.value if hasattr(ops[0].op_type, 'value') else ops[0].op_type
                print_log(f"  {model_name} 金币={state.coins[player]} 选择动作: {get_action_name(op_type)} ({op_type})")
            else:
                print_log(f"  {model_name} choose_operations 返回空列表")
            return ops
        else:
            print_log(f"  {model_name} 没有 choose_bundle 或 choose_operations 方法")
            return []
    except Exception as e:
        print_log(f"✗ 获取 {model_name} 操作失败: {e}")
        sys.path = temp_path
        return []
    finally:
        sys.path = temp_path


def run_detailed_battle(baseline_agent, model_name, model_path, sdk_path, saved_path, max_rounds=200):
    sys.path.insert(0, MAIN_SDK_PATH)
    from SDK.backend.state import PythonBackendState
    from SDK.backend.core import load_backend

    random_ai = BasicRandomAI()
    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=0)
    state = PythonBackendState(game_state)

    round_count = 0
    start_time = time.time()

    print_log(f"\n=== 详细对战开始: {model_name} vs BasicRandomAI ===")
    print_log(f"初始状态: HP(agent)={state.bases[0].hp}, HP(random)={state.bases[1].hp}")

    while not state.terminal and round_count < max_rounds:
        round_count += 1
        print_log(f"\n--- 回合 {round_count} ---")
        print_log(f"当前金币: agent={state.coins[0]}, random={state.coins[1]}")
        print_log(f"当前HP: agent={state.bases[0].hp}, random={state.bases[1].hp}")

        ops_agent = get_baseline_ops(baseline_agent, state, 0, model_path, sdk_path, saved_path, model_name)
        ops_random = random_ai.choose_operations(state, 1)

        state.resolve_turn(ops_agent, ops_random)

        if round_count % 10 == 0:
            print_log(f"回合 {round_count} 结束: HP(agent)={state.bases[0].hp} vs HP(random)={state.bases[1].hp}")

        if state.terminal:
            print_log(f"游戏结束: 胜者={state.winner}")
            break

    duration = time.time() - start_time

    result = 'draw'
    if state.winner == 0:
        result = 'win'
    elif state.winner == 1:
        result = 'loss'

    print_log(f"\n=== 对战结束 ===")
    print_log(f"结果: {result}")
    print_log(f"最终 HP: agent={state.bases[0].hp}, random={state.bases[1].hp}")
    print_log(f"回合数: {round_count}, 耗时: {duration:.2f}s")

    return {
        'result': result,
        'rounds': round_count,
        'duration': duration,
        'final_hp': (state.bases[0].hp, state.bases[1].hp)
    }


def main():
    print_log('=' * 70)
    print_log('详细分析: BasicRandomAI vs Baseline 对战')
    print_log('=' * 70)

    models = ['ann_593']

    for model_name in models:
        print_log(f"\n{'=' * 70}")
        print_log(f"测试模型: {model_name}")
        print_log(f"{'=' * 70}")

        agent_info = load_baseline_model(model_name)
        if agent_info is None:
            continue

        agent, model_path, sdk_path, saved_path = agent_info

        run_detailed_battle(agent, model_name, model_path, sdk_path, saved_path, max_rounds=100)


if __name__ == '__main__':
    main()