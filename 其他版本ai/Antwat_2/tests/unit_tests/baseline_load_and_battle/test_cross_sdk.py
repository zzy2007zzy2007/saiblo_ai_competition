#!/usr/bin/env python3
"""测试：项目SDK的BasicRandomAI vs 自己SDK的baseline模型"""
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


class BasicRandomAI:
    """使用项目SDK的随机AI"""

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
                return list(selected.operations)
        except Exception as e:
            print_log(f"BasicRandomAI 错误: {e}")
        return []


def load_baseline_model(model_name):
    """加载baseline模型，使用自己的SDK"""
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
        print_log(f"✓ 加载 {model_name} AI 成功（使用自己的SDK）")
        sys.path = original_path
        return agent, model_path, sdk_path, original_path
    except Exception as e:
        print_log(f"✗ 加载失败: {e}")
        sys.path = original_path
        return None


def get_baseline_ops(agent, state, player, model_path, sdk_path, saved_path):
    """使用baseline自己的SDK获取操作"""
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
                return ops
        elif hasattr(agent, 'choose_operations'):
            ops = agent.choose_operations(state, player)
            return ops
        return []
    except Exception as e:
        print_log(f"获取 baseline 操作失败: {e}")
        return []
    finally:
        sys.path = temp_path


def run_battle(baseline_agent, model_name, model_path, sdk_path, saved_path, max_rounds=512):
    """运行对战：BasicRandomAI(项目SDK) vs baseline(自己SDK)"""
    random_ai = BasicRandomAI()

    # 使用baseline的SDK创建游戏状态
    temp_path = sys.path.copy()
    try:
        sys.path = []
        sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)
        for p in temp_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)

        from SDK.backend.core import load_backend
        from SDK.backend.state import PythonBackendState

        backend = load_backend(prefer_native=False)
        game_state = backend.initial_state(seed=0)
        state = PythonBackendState(game_state)
    finally:
        sys.path = temp_path

    round_count = 0
    start_time = time.time()

    print_log(f"\n=== 对战开始: {model_name}(自己SDK) vs BasicRandomAI(项目SDK) ===")
    print_log(f"初始状态: HP({model_name})={state.bases[0].hp}, HP(random)={state.bases[1].hp}")

    while not state.terminal and round_count < max_rounds:
        round_count += 1

        ops_agent = get_baseline_ops(baseline_agent, state, 0, model_path, sdk_path, saved_path)
        ops_random = random_ai.choose_operations(state, 1)

        # 使用baseline的SDK执行回合
        temp_path2 = sys.path.copy()
        try:
            sys.path = []
            sys.path.insert(0, sdk_path)
            sys.path.insert(0, model_path)
            for p in temp_path2:
                if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                    sys.path.append(p)
            state.resolve_turn(ops_agent, ops_random)
        finally:
            sys.path = temp_path2

        if round_count % 50 == 0 or state.terminal:
            print_log(f"回合 {round_count}: HP({model_name})={state.bases[0].hp} vs HP(random)={state.bases[1].hp}")

        if state.terminal:
            break

    duration = time.time() - start_time

    result = 'draw'
    if state.winner == 0:
        result = 'win'
    elif state.winner == 1:
        result = 'loss'

    print_log(f"\n=== 对战结束 ===")
    print_log(f"结果: {result}")
    print_log(f"最终 HP: {model_name}={state.bases[0].hp}, random={state.bases[1].hp}")
    print_log(f"回合数: {round_count}, 耗时: {duration:.2f}s")

    return {
        'result': result,
        'rounds': round_count,
        'duration': duration,
        'final_hp': (state.bases[0].hp, state.bases[1].hp)
    }


def main():
    print_log('=' * 70)
    print_log('测试: 项目SDK的BasicRandomAI vs 自己SDK的baseline模型')
    print_log('=' * 70)

    models = ['gen99', 'gen199', 'ann_593']
    all_results = {}

    for model_name in models:
        print_log(f"\n{'=' * 70}")
        print_log(f"测试模型: {model_name}")
        print_log(f"{'=' * 70}")

        agent_info = load_baseline_model(model_name)
        if agent_info is None:
            continue

        agent, model_path, sdk_path, saved_path = agent_info

        results = {'wins': 0, 'losses': 0, 'draws': 0}
        for episode in range(2):
            print_log(f"\n--- Episode {episode + 1} ---")
            battle_result = run_battle(agent, model_name, model_path, sdk_path, saved_path, max_rounds=512)
            if battle_result['result'] == 'win':
                results['wins'] += 1
            elif battle_result['result'] == 'loss':
                results['losses'] += 1
            else:
                results['draws'] += 1

        all_results[model_name] = results
        win_rate = results['wins'] / 2 * 100 if 2 > 0 else 0
        print_log(f"\n{model_name} 汇总: {results['wins']}W/{results['losses']}L/{results['draws']}D, 胜率: {win_rate:.1f}%")

    print_log(f"\n{'=' * 70}")
    print_log("最终汇总")
    print_log(f"{'=' * 70}")
    for model_name, result in all_results.items():
        win_rate = result['wins'] / 2 * 100 if 2 > 0 else 0
        print_log(f"{model_name}: {result['wins']}W/{result['losses']}L/{result['draws']}D, 胜率: {win_rate:.1f}%")


if __name__ == '__main__':
    main()