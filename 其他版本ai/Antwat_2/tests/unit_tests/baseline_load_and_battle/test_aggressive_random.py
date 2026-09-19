#!/usr/bin/env python3
"""修改 BasicRandomAI，使其更积极地攻击"""
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


class AggressiveRandomAI:
    """更积极的随机AI，倾向于选择攻击相关的动作"""

    def __init__(self):
        self.sdk_path = MAIN_SDK_PATH
        sys.path.insert(0, self.sdk_path)
        from SDK.utils.actions import ActionCatalog
        self.ActionCatalog = ActionCatalog
        self.attack_action_types = {1, 2, 4, 5, 10}  # 攻击/建造相关动作

    def choose_operations(self, state, player):
        try:
            catalog = self.ActionCatalog(max_actions=32)
            bundles = catalog.build(state, player)
            
            if not bundles or len(bundles) == 0:
                return []

            # 分离攻击动作和非攻击动作
            attack_bundles = []
            other_bundles = []
            
            for bundle in bundles:
                if bundle.operations:
                    op_type = bundle.operations[0].op_type.value if hasattr(bundle.operations[0].op_type, 'value') else bundle.operations[0].op_type
                    if op_type in self.attack_action_types:
                        attack_bundles.append(bundle)
                    else:
                        other_bundles.append(bundle)

            # 如果有攻击动作，70%概率选择攻击动作
            if attack_bundles:
                if random.random() < 0.7:
                    selected = random.choice(attack_bundles)
                else:
                    selected = random.choice(other_bundles)
            else:
                selected = random.choice(other_bundles)

            ops = list(selected.operations)
            if ops:
                op_type = ops[0].op_type.value if hasattr(ops[0].op_type, 'value') else ops[0].op_type
                print_log(f"  RandomAI 金币={state.coins[player]} 选择动作: {get_action_name(op_type)} ({op_type})")
            return ops
        except Exception as e:
            print_log(f"AggressiveRandomAI 错误: {e}")
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
                print_log(f"  {model_name} 选择动作: {get_action_name(op_type)} ({op_type})")
            return ops
        return []
    except Exception as e:
        print_log(f"✗ 获取 {model_name} 操作失败: {e}")
        return []
    finally:
        sys.path = temp_path


def run_battle(baseline_agent, model_name, model_path, sdk_path, saved_path, max_rounds=100):
    sys.path.insert(0, MAIN_SDK_PATH)
    from SDK.backend.state import PythonBackendState
    from SDK.backend.core import load_backend

    aggressive_ai = AggressiveRandomAI()
    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=0)
    state = PythonBackendState(game_state)

    round_count = 0
    start_time = time.time()

    print_log(f"\n=== 对战开始: {model_name} vs AggressiveRandomAI ===")
    print_log(f"初始状态: HP(agent)={state.bases[0].hp}, HP(random)={state.bases[1].hp}")

    while not state.terminal and round_count < max_rounds:
        round_count += 1

        ops_agent = get_baseline_ops(baseline_agent, state, 0, model_path, sdk_path, saved_path, model_name)
        ops_random = aggressive_ai.choose_operations(state, 1)

        state.resolve_turn(ops_agent, ops_random)

        if round_count % 10 == 0 or state.terminal:
            print_log(f"回合 {round_count}: HP(agent)={state.bases[0].hp} vs HP(random)={state.bases[1].hp}")

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
    print_log('AggressiveRandomAI vs Baseline 对战测试')
    print_log('=' * 70)

    models = ['ann_593', 'gen99', 'gen199']
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
            battle_result = run_battle(agent, model_name, model_path, sdk_path, saved_path, max_rounds=100)
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