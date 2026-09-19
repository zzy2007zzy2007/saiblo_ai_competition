#!/usr/bin/env python3
"""
简化版测试 baseline 模型与 BasicTowerAI 的对战
- BasicTowerAI 使用项目主 SDK
- 其他 baseline 模型使用自己的 SDK
"""

import os
import sys
import time
from datetime import datetime


def print_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f"[{timestamp}] {message}")


def load_agent_with_own_sdk(base_dir, model_name):
    """加载 agent 并设置其 SDK"""
    model_path = os.path.join(base_dir, "baselines", model_name)
    sdk_path = os.path.join(model_path, "SDK")

    if not os.path.exists(model_path):
        print_log(f"✗ 模型路径不存在: {model_path}")
        return None

    original_sys_path = sys.path.copy()
    new_path = []

    if os.path.exists(sdk_path):
        new_path.insert(0, sdk_path)

    new_path.insert(0, model_path)

    for p in original_sys_path:
        if "site-packages" in p or p == "" or "python3.12" in p or "lib/python" in p:
            new_path.append(p)

    sys.path = new_path

    try:
        ai_path = os.path.join(model_path, "ai.py")
        if os.path.exists(ai_path):
            from ai import AI
            agent = AI()
            print_log(f"✓ 加载 AI: {model_name}")
            sys.path = original_sys_path
            return agent, model_path, sdk_path
        else:
            print_log(f"✗ 找不到 ai.py: {model_name}")
            sys.path = original_sys_path
            return None
    except Exception as e:
        import traceback
        print_log(f"✗ 加载失败: {model_name}: {e}")
        print_log(traceback.format_exc())
        sys.path = original_sys_path
        return None


def load_main_sdk(base_dir):
    """加载主 SDK（用于 BasicTowerAI）"""
    antwar_path = os.path.join(base_dir, "saiblo-antwar-sdk-python")
    if os.path.exists(antwar_path):
        sys.path.insert(0, antwar_path)
    print_log(f"✓ 加载主 SDK: {antwar_path}")


def create_basic_tower_ai():
    """创建 BasicTowerAI（使用主 SDK）"""
    from SDK.backend.model import Operation as BackendOperation
    from SDK.utils.constants import OperationType as BackendOperationType
    from antwar.coord import Coord
    from antwar.protocol import build_tower_op

    class BasicTowerAI:
        def choose_operations(self, state, player):
            positions = [Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12)]
            for pos in positions:
                if state.coins[player] >= 30:
                    op = build_tower_op(pos)
                    return [BackendOperation(
                        op_type=BackendOperationType(op.type.value),
                        arg0=op.arg0, arg1=op.arg1
                    )]
            return []

    return BasicTowerAI()


def get_agent_operations_with_sdk(agent, state, player, model_path, sdk_path):
    """使用指定 SDK 获取操作"""
    original_sys_path = sys.path.copy()

    try:
        new_path = []
        if sdk_path:
            new_path.insert(0, sdk_path)
        new_path.insert(0, model_path)
        for p in original_sys_path:
            if "site-packages" in p or p == "" or "python3.12" in p or "lib/python" in p:
                new_path.append(p)
        sys.path = new_path

        result = agent.choose_operations(state, player)
        sys.path = original_sys_path
        return result
    except Exception as e:
        import traceback
        print_log(f"✗ 获取操作失败: {e}")
        sys.path = original_sys_path
        return []


def run_single_battle(agent, model_name, model_path, sdk_path, opponent, episode, max_rounds):
    """运行单场对战"""
    from SDK.backend.state import PythonBackendState
    from SDK.backend.core import load_backend

    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=episode)
    state = PythonBackendState(game_state)

    round_count = 0
    start_time = time.time()

    while not state.terminal and round_count < max_rounds:
        round_count += 1

        try:
            ops_agent = get_agent_operations_with_sdk(agent, state, 0, model_path, sdk_path)
        except Exception as e:
            print_log(f"✗ 获取 {model_name} 操作失败: {e}")
            ops_agent = []

        try:
            ops_opponent = opponent.choose_operations(state, 1)
        except Exception as e:
            print_log(f"✗ 获取 BasicTowerAI 操作失败: {e}")
            ops_opponent = []

        state.resolve_turn(ops_agent, ops_opponent)

        if round_count % 20 == 0:
            hp_info = f"HP: {model_name}={state.bases[0].hp}, BasicTowerAI={state.bases[1].hp}"
            print_log(f"回合 {round_count}: {hp_info}")

    result = 'draw'
    if state.winner == 0:
        result = 'win'
    elif state.winner == 1:
        result = 'loss'

    duration = time.time() - start_time
    return {
        'result': result,
        'rounds_played': round_count,
        'duration': duration,
        'final_hp': {'agent': state.bases[0].hp, 'opponent': state.bases[1].hp}
    }


def test_model(base_dir, model_name, episodes=2, max_rounds=100):
    """测试单个模型"""
    print_log(f"\n{'=' * 70}")
    print_log(f"测试模型: {model_name}")
    print_log(f"{'=' * 70}")

    agent_info = load_agent_with_own_sdk(base_dir, model_name)
    if agent_info is None:
        print_log(f"✗ 跳过 {model_name} 加载失败")
        return None

    agent, model_path, sdk_path = agent_info
    opponent = create_basic_tower_ai()

    agent_wins = 0
    basic_wins = 0
    draws = 0
    total_rounds = 0

    for episode in range(episodes):
        print_log(f"\n--- 第 {episode + 1} 场开始 ---")

        battle_result = run_single_battle(
            agent, model_name, model_path, sdk_path,
            opponent, episode, max_rounds
        )

        total_rounds += battle_result['rounds_played']

        if battle_result['result'] == 'win':
            agent_wins += 1
            print_log(f"✓ 第 {episode + 1} 场: {model_name} 胜")
        elif battle_result['result'] == 'loss':
            basic_wins += 1
            print_log(f"✗ 第 {episode + 1} 场: {model_name} 负")
        else:
            draws += 1
            print_log(f"— 第 {episode + 1} 场: 平局")

        print_log(f"最终HP: {model_name}={battle_result['final_hp']['agent']}, "
                  f"BasicTowerAI={battle_result['final_hp']['opponent']}")
        print_log(f"回合数: {battle_result['rounds_played']}, 耗时: {battle_result['duration']:.2f}s")

    win_rate = agent_wins / episodes if episodes > 0 else 0

    print_log(f"\n{'=' * 70}")
    print_log(f"{model_name} vs BasicTowerAI")
    print_log(f"{'=' * 70}")
    print_log(f"胜场: {agent_wins}")
    print_log(f"负场: {basic_wins}")
    print_log(f"平局: {draws}")
    print_log(f"胜率: {win_rate * 100:.1f}%")
    print_log(f"总回合数: {total_rounds}")

    return {
        'model_name': model_name,
        'agent_wins': agent_wins,
        'basic_wins': basic_wins,
        'draws': draws,
        'win_rate': win_rate,
        'total_rounds': total_rounds
    }


def main():
    base_dir = '/root/autodl-tmp/AntWar'

    print_log("=" * 70)
    print_log("Baseline 模型对战测试")
    print_log("=" * 70)

    load_main_sdk(base_dir)

    models = ["ann_593", "gen99", "gen199"]
    results = {}

    for model_name in models:
        result = test_model(base_dir, model_name, episodes=2, max_rounds=100)
        if result:
            results[model_name] = result

    print_log(f"\n{'=' * 70}")
    print_log("最终汇总")
    print_log(f"{'=' * 70}")

    for model_name, result in results.items():
        print_log(f"{model_name}: {result['agent_wins']}胜/{result['basic_wins']}负/"
                  f"{result['draws']}平, 胜率 {result['win_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
