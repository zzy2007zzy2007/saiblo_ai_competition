#!/usr/bin/env python3
"""详细检查蚂蚁移动和战斗"""

import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from SDK.backend.model import Operation as BackendOperation
from SDK.utils.constants import OperationType as BackendOperationType

backend = load_backend(prefer_native=False)
game_state = backend.initial_state(seed=0)
state = PythonBackendState(game_state)

print("=== 建造塔并等待蚂蚁 ===\n")

op = BackendOperation(op_type=BackendOperationType.BUILD_TOWER, arg0=4, arg1=6)
state.apply_operation(0, op)
state.advance_round()

op2 = BackendOperation(op_type=BackendOperationType.BUILD_TOWER, arg0=13, arg1=6)
state.apply_operation(1, op2)
state.advance_round()

print("建造完成，等待回合...\n")

for i in range(30):
    state.advance_round()

    p0_ants = [a for a in state.ants if a.player == 0]
    p1_ants = [a for a in state.ants if a.player == 1]

    print(f"回合 {i+1}: P0蚂蚁数={len(p0_ants)}, P1蚂蚁数={len(p1_ants)}, HP: P0={state.bases[0].hp}, P1={state.bases[1].hp}")

    if p0_ants:
        a = p0_ants[0]
        print(f"  P0蚂蚁[0]: ({a.x}, {a.y}) HP={a.hp}")

    if p1_ants:
        a = p1_ants[0]
        print(f"  P1蚂蚁[0]: ({a.x}, {a.y}) HP={a.hp}")

    if state.bases[0].hp < 50 or state.bases[1].hp < 50:
        print("\n*** 基座受伤! ***")
        break

    if i >= 20:
        break

print("\n=== 结论 ===")
print(f"30回合后：P0基座HP={state.bases[0].hp}, P1基座HP={state.bases[1].hp}")
print(f"蚂蚁数量：P0={len([a for a in state.ants if a.player==0])}, P1={len([a for a in state.ants if a.player==1])}")
print(f"塔数量：{len(state.towers)}")