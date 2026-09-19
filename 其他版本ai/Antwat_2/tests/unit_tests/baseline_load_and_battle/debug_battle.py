#!/usr/bin/env python3
"""对战调试脚本"""

import os
import sys

sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/gen99')

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from SDK.backend.model import Operation as BackendOperation
from SDK.utils.constants import OperationType as BackendOperationType
from SDK.utils.actions import ActionBundle

class BasicTowerAI:
    def choose_operations(self, state, player):
        from antwar.coord import Coord
        positions = [Coord(4, 6), Coord(4, 12), Coord(13, 6), Coord(13, 12)]
        for pos in positions:
            if state.coins[player] >= 30:
                from antwar.protocol import build_tower_op
                return [build_tower_op(pos)]
        return []

def test_battle():
    print("=" * 60)
    print("对战调试测试")
    print("=" * 60)

    from ai import AI as RuleBasedAI
    agent = RuleBasedAI()

    basic_tower = BasicTowerAI()

    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=0)
    state = PythonBackendState(game_state)

    print(f"\n初始状态:")
    print(f"  Player 0 (我方) - 金币: {state.coins[0]}, 基座 HP: {state.bases[0].hp}")
    print(f"  Player 1 (敌方) - 金币: {state.coins[1]}, 基座 HP: {state.bases[1].hp}")

    for round_count in range(1, 11):
        print(f"\n=== 回合 {round_count} ===")
        print(f"回合开始 - P0金币:{state.coins[0]}, P1金币:{state.coins[1]}")

        bundles = agent.list_bundles(state, 0)
        print(f"  我方候选动作数: {len(bundles)}")
        if bundles:
            best_bundle = agent.choose_bundle(state, 0, bundles)
            print(f"  我方选择: score={best_bundle.score}, ops={len(best_bundle.operations)}")
            for op in best_bundle.operations:
                print(f"    SDK.Operation: op_type={op.op_type}, arg0={op.arg0}, arg1={op.arg1}")

            for op in best_bundle.operations:
                try:
                    state.apply_operation(0, op)
                    print(f"    执行成功")
                except Exception as e:
                    print(f"    执行失败: {e}")

        ops = basic_tower.choose_operations(state, 1)
        print(f"  敌方选择 {len(ops)} 个动作")
        for op in ops:
            print(f"    antwar.Operation: type={op.type}, arg0={op.arg0}, arg1={op.arg1}")
            try:
                backend_op = BackendOperation(
                    op_type=BackendOperationType(op.type.value),
                    arg0=op.arg0, arg1=op.arg1
                )
                state.apply_operation(1, backend_op)
                print(f"    执行成功")
            except Exception as e:
                print(f"    执行失败: {e}")

        state.advance_round()

        print(f"回合结束 - P0金币:{state.coins[0]}, P1金币:{state.coins[1]}")
        print(f"          P0基座HP:{state.bases[0].hp}, P1基座HP:{state.bases[1].hp}")

        if state.bases[0].hp <= 0 or state.bases[1].hp <= 0:
            print(f"\n游戏结束!")
            break

    print(f"\n最终状态:")
    print(f"  Player 0 - 基座 HP: {state.bases[0].hp}")
    print(f"  Player 1 - 基座 HP: {state.bases[1].hp}")

if __name__ == "__main__":
    test_battle()