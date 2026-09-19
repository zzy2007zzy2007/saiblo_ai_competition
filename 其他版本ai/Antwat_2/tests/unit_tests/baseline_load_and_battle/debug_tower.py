#!/usr/bin/env python3
"""对战调试脚本 - 检查塔的行为"""

import os
import sys

sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/gen99')

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from SDK.backend.model import Operation as BackendOperation
from SDK.utils.constants import OperationType as BackendOperationType

def test_tower_behavior():
    print("=" * 60)
    print("塔行为调试测试")
    print("=" * 60)

    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=0)
    state = PythonBackendState(game_state)

    print(f"\n初始状态:")
    print(f"  Player 0 - 金币: {state.coins[0]}, 基座 HP: {state.bases[0].hp}")
    print(f"  Player 1 - 金币: {state.coins[1]}, 基座 HP: {state.bases[1].hp}")

    print("\n--- 在 (4,6) 建造塔 (P0) ---")
    op1 = BackendOperation(op_type=BackendOperationType.BUILD_TOWER, arg0=4, arg1=6)
    print(f"执行: BUILD_TOWER at (4, 6)")
    state.apply_operation(0, op1)
    state.advance_round()

    print(f"\n建造后 P0金币: {state.coins[0]}")
    print(f"塔数量: {len(state.towers)}")

    print("\n--- 在 (13,6) 建造塔 (P1) ---")
    op2 = BackendOperation(op_type=BackendOperationType.BUILD_TOWER, arg0=13, arg1=6)
    print(f"执行: BUILD_TOWER at (13, 6)")
    state.apply_operation(1, op2)
    state.advance_round()

    print(f"\n建造后 P1金币: {state.coins[1]}")
    print(f"塔数量: {len(state.towers)}")

    print("\n--- 等待几个回合让塔攻击 ---")
    for i in range(5):
        state.advance_round()
        print(f"回合 {i+1} 后: P0 HP={state.bases[0].hp}, P1 HP={state.bases[1].hp}")

    print("\n--- 建造更多塔 ---")
    positions = [(4, 12), (13, 12)]
    for x, y in positions:
        op = BackendOperation(op_type=BackendOperationType.BUILD_TOWER, arg0=x, arg1=y)
        player = 0 if x < 9 else 1
        print(f"P{player} 建造塔 at ({x}, {y})")
        state.apply_operation(player, op)
        state.advance_round()

    print(f"\n当前状态: P0 HP={state.bases[0].hp}, P1 HP={state.bases[1].hp}")
    print(f"塔数量: {len(state.towers)}")

    print("\n--- 等待更多回合 ---")
    for i in range(10):
        state.advance_round()
        if state.bases[0].hp <= 0 or state.bases[1].hp <= 0:
            print(f"回合 {i+1}: P0 HP={state.bases[0].hp}, P1 HP={state.bases[1].hp} - 游戏结束!")
            break
        if i % 3 == 0:
            print(f"回合 {i+1}: P0 HP={state.bases[0].hp}, P1 HP={state.bases[1].hp}")

if __name__ == "__main__":
    test_tower_behavior()