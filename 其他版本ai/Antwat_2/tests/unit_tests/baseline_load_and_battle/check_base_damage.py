#!/usr/bin/env python3
"""测试蚂蚁到达对方基地时是否造成伤害"""

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

print("=== 游戏初始状态 ===")
print(f"P0 基座位置: ({state.bases[0].x}, {state.bases[0].y}), HP: {state.bases[0].hp}")
print(f"P1 基座位置: ({state.bases[1].x}, {state.bases[1].y}), HP: {state.bases[1].hp}")

print("\n=== 不建造任何塔，观察蚂蚁自然移动 ===")

for i in range(50):
    state.advance_round()

    p0_ants = [a for a in state.ants if a.player == 0]
    p1_ants = [a for a in state.ants if a.player == 1]

    if i < 20 or i % 5 == 0:
        print(f"回合 {i+1}: P0蚂蚁={len(p0_ants)}, P1蚂蚁={len(p1_ants)}, HP: P0={state.bases[0].hp}, P1={state.bases[1].hp}")

    if state.bases[0].hp < 50 or state.bases[1].hp < 50:
        print(f"\n*** 基座HP变化! P0={state.bases[0].hp}, P1={state.bases[1].hp} ***")
        break

    if len(state.ants) > 50:
        print("\n太多蚂蚁，停止")
        break

print("\n=== 检查 P0 蚂蚁位置 ===")
p0_ants = [a for a in state.ants if a.player == 0][:5]
for ant in p0_ants:
    print(f"  蚂蚁{ant.ant_id}: ({ant.x}, {ant.y}), HP={ant.hp}")

print("\n=== 检查 P1 蚂蚁位置 ===")
p1_ants = [a for a in state.ants if a.player == 1][:5]
for ant in p1_ants:
    print(f"  蚂蚁{ant.ant_id}: ({ant.x}, {ant.y}), HP={ant.hp}")