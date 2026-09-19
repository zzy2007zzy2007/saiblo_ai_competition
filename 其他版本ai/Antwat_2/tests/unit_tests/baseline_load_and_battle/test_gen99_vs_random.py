#!/usr/bin/env python3
"""测试：gen99(基于规则) vs BasicRandomAI(基于规则)"""
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

    def choose_bundle(self, state, player, bundles=None):
        try:
            bundles = bundles or self.list_bundles(state, player)
            if bundles and len(bundles) > 0:
                selected = random.choice(bundles)
                return selected
        except Exception as e:
            print_log(f"BasicRandomAI 错误: {e}")
        return None

    def list_bundles(self, state, player):
        catalog = self.ActionCatalog(max_actions=32)
        return catalog.build(state, player)


def load_gen99():
    """加载gen99，使用自己的SDK"""
    model_path = os.path.join(BASE_DIR, 'baselines', 'gen99')
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
        print_log(f"✓ 成功加载 gen99 (基于规则，使用自己的SDK)")
        sys.path = original_path
        return agent, model_path, sdk_path, original_path
    except Exception as e:
        print_log(f"✗ 加载 gen99 失败: {e}")
        sys.path = original_path
        return None


def run_battle(gen99_agent, model_path, sdk_path, saved_path, max_rounds=512):
    """运行对战：gen99(自己SDK) vs BasicRandomAI(项目SDK)"""
    random_ai = BasicRandomAI()

    # 使用gen99的SDK创建游戏状态
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

    print_log(f"\n=== 对战开始: gen99(规则) vs BasicRandomAI(随机) ===")
    print_log(f"初始状态: HP(gen99)={state.bases[0].hp}, HP(random)={state.bases[1].hp}")

    while not state.terminal and round_count < max_rounds:
        round_count += 1

        # gen99 使用自己的SDK
        temp_path2 = sys.path.copy()
        try:
            sys.path = []
            sys.path.insert(0, sdk_path)
            sys.path.insert(0, model_path)
            for p in temp_path2:
                if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                    sys.path.append(p)

            bundles0 = gen99_agent.list_bundles(state, 0)
            bundle0 = gen99_agent.choose_bundle(state, 0, bundles0)
            ops0 = list(bundle0.operations) if bundle0 else []
        finally:
            sys.path = temp_path2

        # BasicRandomAI 使用项目SDK
        bundles1 = random_ai.list_bundles(state, 1)
        bundle1 = random_ai.choose_bundle(state, 1, bundles1)
        ops1 = list(bundle1.operations) if bundle1 else []

        # 使用gen99的SDK执行回合
        temp_path3 = sys.path.copy()
        try:
            sys.path = []
            sys.path.insert(0, sdk_path)
            sys.path.insert(0, model_path)
            for p in temp_path3:
                if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                    sys.path.append(p)
            state.resolve_turn(ops0, ops1)
        finally:
            sys.path = temp_path3

        if round_count % 50 == 0 or state.terminal:
            print_log(f"回合 {round_count}: HP(gen99)={state.bases[0].hp} vs HP(random)={state.bases[1].hp}")

        if state.terminal:
            break

    duration = time.time() - start_time

    result = 'draw'
    winner_name = 'Draw'
    if state.winner == 0:
        result = 'win'
        winner_name = 'gen99'
    elif state.winner == 1:
        result = 'loss'
        winner_name = 'BasicRandomAI'

    print_log(f"\n=== 对战结束 ===")
    print_log(f"结果: {result} ({winner_name} 获胜)")
    print_log(f"最终 HP: gen99={state.bases[0].hp}, random={state.bases[1].hp}")
    print_log(f"回合数: {round_count}, 耗时: {duration:.2f}s")

    return {
        'result': result,
        'rounds': round_count,
        'duration': duration,
        'final_hp': (state.bases[0].hp, state.bases[1].hp)
    }


def main():
    print_log('=' * 70)
    print_log('测试: gen99(基于规则) vs BasicRandomAI(随机)')
    print_log('=' * 70)

    agent_info = load_gen99()
    if agent_info is None:
        print_log("✗ 无法加载 gen99，测试终止")
        return

    agent, model_path, sdk_path, saved_path = agent_info

    results = {'wins': 0, 'losses': 0, 'draws': 0}
    for episode in range(3):
        print_log(f"\n--- Episode {episode + 1} ---")
        battle_result = run_battle(agent, model_path, sdk_path, saved_path, max_rounds=512)
        if battle_result['result'] == 'win':
            results['wins'] += 1
        elif battle_result['result'] == 'loss':
            results['losses'] += 1
        else:
            results['draws'] += 1

    win_rate = results['wins'] / 3 * 100 if 3 > 0 else 0
    print_log(f"\n{'=' * 70}")
    print_log("最终汇总")
    print_log(f"{'=' * 70}")
    print_log(f"gen99 vs BasicRandomAI: {results['wins']}W/{results['losses']}L/{results['draws']}D")
    print_log(f"gen99 胜率: {win_rate:.1f}%")


if __name__ == '__main__':
    main()