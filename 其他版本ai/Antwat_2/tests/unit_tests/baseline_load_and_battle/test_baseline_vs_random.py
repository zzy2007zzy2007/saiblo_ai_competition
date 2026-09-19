#!/usr/bin/env python3
"""测试脚本：BasicRandomAI vs Baseline模型

测试目标：
1. BasicRandomAI 使用项目 SDK
2. baseline 模型使用各自的 SDK
3. 各对战 2 局，观察是否有问题
"""

import os
import sys
import time
import random
from datetime import datetime


BASE_DIR = '/root/autodl-tmp/AntWar'
MAIN_SDK_PATH = os.path.join(BASE_DIR, 'ppo_v1', 'Ant-Game', 'SDK')


def print_log(message):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")


class BasicRandomAI:
    """随机AI：每次从合法动作中随机选择一个执行"""

    def __init__(self):
        self.sdk_path = MAIN_SDK_PATH
        self._load_modules()

    def _load_modules(self):
        """加载必要的模块"""
        sys.path.insert(0, self.sdk_path)
        from SDK.backend.state import PythonBackendState
        from SDK.backend.core import load_backend
        from SDK.backend.model import Operation as BackendOperation
        from SDK.utils.constants import OperationType as BackendOperationType
        from SDK.utils.actions import ActionCatalog

        self.PythonBackendState = PythonBackendState
        self.load_backend = load_backend
        self.BackendOperation = BackendOperation
        self.OperationType = BackendOperationType
        self.ActionCatalog = ActionCatalog

    def _get_valid_actions(self, state, player):
        """获取当前玩家的所有合法动作"""
        try:
            catalog = self.ActionCatalog(max_actions=32)
            bundles = catalog.build(state, player)
            return bundles
        except Exception as e:
            print_log(f"获取动作失败: {e}")
            return []

    def choose_operations(self, state, player):
        """随机选择一个合法动作"""
        try:
            bundles = self._get_valid_actions(state, player)
            if bundles and len(bundles) > 0:
                selected = random.choice(bundles)
                return list(selected.operations)
        except Exception as e:
            print_log(f"BasicRandomAI choose_operations 错误: {e}")
        return []


def load_baseline_model(model_name):
    """加载 baseline 模型，使用模型自己的 SDK"""
    model_path = os.path.join(BASE_DIR, 'baselines', model_name)
    sdk_path = os.path.join(model_path, 'SDK')

    if not os.path.exists(model_path):
        print_log(f"✗ 模型路径不存在: {model_name}")
        return None
    if not os.path.exists(sdk_path):
        print_log(f"✗ SDK路径不存在: {sdk_path}")
        return None

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
        import traceback
        print_log(f"✗ 加载 {model_name} 失败: {e}")
        sys.path = original_path
        return None


def get_baseline_ops(Agent, state, player, model_path, sdk_path, saved_path):
    """使用 baseline 模型的 SDK 获取操作"""
    temp_path = sys.path.copy()
    try:
        sys.path = []
        sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)

        for p in saved_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)

        if hasattr(Agent, 'choose_bundle') and hasattr(Agent, 'list_bundles'):
            bundles = Agent.list_bundles(state, player)
            if bundles:
                bundle = Agent.choose_bundle(state, player, bundles)
                ops = list(bundle.operations)
            else:
                ops = []
        elif hasattr(Agent, 'choose_operations'):
            ops = Agent.choose_operations(state, player)
        else:
            ops = []

        sys.path = temp_path
        return ops
    except Exception as e:
        print_log(f"获取 baseline 操作失败: {e}")
        sys.path = temp_path
        return []


def run_battle(baseline_agent, model_name, model_path, sdk_path, saved_path, episode, max_rounds=100):
    """运行单场对战"""
    sys.path.insert(0, MAIN_SDK_PATH)
    from SDK.backend.state import PythonBackendState
    from SDK.backend.core import load_backend

    random_ai = BasicRandomAI()

    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=episode)
    state = PythonBackendState(game_state)

    round_count = 0
    start_time = time.time()

    while not state.terminal and round_count < max_rounds:
        round_count += 1

        ops_agent = get_baseline_ops(baseline_agent, state, 0, model_path, sdk_path, saved_path)
        ops_random = random_ai.choose_operations(state, 1)

        state.resolve_turn(ops_agent, ops_random)

        if round_count % 20 == 0:
            print_log(f"Round {round_count}: HP(agent)={state.bases[0].hp} vs HP(random)={state.bases[1].hp}")

    duration = time.time() - start_time

    result = 'draw'
    if state.winner == 0:
        result = 'win'
    elif state.winner == 1:
        result = 'loss'

    return {
        'result': result,
        'rounds': round_count,
        'duration': duration,
        'final_hp': (state.bases[0].hp, state.bases[1].hp)
    }


def test_baseline_vs_random(model_name, episodes=2):
    """测试单个 baseline 模型"""
    print_log(f"\n{'=' * 70}")
    print_log(f"测试模型: {model_name}")
    print_log(f"{'=' * 70}")

    agent_info = load_baseline_model(model_name)
    if agent_info is None:
        return None

    Agent, model_path, sdk_path, saved_path = agent_info

    results = {
        'wins': 0, 'losses': 0, 'draws': 0,
        'total_rounds': 0, 'total_duration': 0
    }

    for episode in range(episodes):
        print_log(f"\n--- {model_name} vs BasicRandomAI - Episode {episode + 1} ---")

        battle_result = run_battle(
            Agent, model_name, model_path, sdk_path, saved_path, episode
        )

        results['total_rounds'] += battle_result['rounds']
        results['total_duration'] += battle_result['duration']

        if battle_result['result'] == 'win':
            results['wins'] += 1
            print_log(f"✓ Episode {episode + 1}: WIN")
        elif battle_result['result'] == 'loss':
            results['losses'] += 1
            print_log(f"✗ Episode {episode + 1}: LOSS")
        else:
            results['draws'] += 1
            print_log(f"— Episode {episode + 1}: DRAW")

        print_log(f"  Final HP: {battle_result['final_hp'][0]} vs {battle_result['final_hp'][1]}")
        print_log(f"  Rounds: {battle_result['rounds']}, Duration: {battle_result['duration']:.2f}s")

    win_rate = results['wins'] / episodes * 100 if episodes > 0 else 0
    print_log(f"\n{'=' * 70}")
    print_log(f"{model_name} vs BasicRandomAI 汇总")
    print_log(f"{'=' * 70}")
    print_log(f"胜场: {results['wins']}, 负场: {results['losses']}, 平局: {results['draws']}")
    print_log(f"胜率: {win_rate:.1f}%")
    print_log(f"总回合数: {results['total_rounds']}, 总耗时: {results['total_duration']:.2f}s")

    return results


def main():
    print_log('=' * 70)
    print_log('Baseline 模型 vs BasicRandomAI 对战测试')
    print_log('=' * 70)
    print_log(f"项目 SDK: {MAIN_SDK_PATH}")

    models = ['ann_593', 'gen99', 'gen199']
    all_results = {}

    for model_name in models:
        result = test_baseline_vs_random(model_name, episodes=2)
        if result:
            all_results[model_name] = result

    print_log(f"\n{'=' * 70}")
    print_log("最终汇总 - 所有模型")
    print_log(f"{'=' * 70}")

    for model_name, result in all_results.items():
        win_rate = result['wins'] / 2 * 100 if 2 > 0 else 0
        print_log(f"{model_name}: {result['wins']}W/{result['losses']}L/{result['draws']}D, 胜率: {win_rate:.1f}%")


if __name__ == '__main__':
    main()