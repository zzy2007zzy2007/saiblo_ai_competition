#!/usr/bin/env python3
"""检查蚂蚁和塔的攻击行为"""

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

print("初始蚂蚁数量:", len(state.ants))
print("初始塔数量:", len(state.towers))

print("\n=== 建造塔 ===")
op = BackendOperation(op_type=BackendOperationType.BUILD_TOWER, arg0=4, arg1=6)
state.apply_operation(0, op)
state.advance_round()

print(f"塔数量: {len(state.towers)}")
print(f"蚂蚁数量: {len(state.ants)}")

print("\n=== 等待回合，观察蚂蚁生成 ===")
for i in range(10):
    state.advance_round()
    print(f"回合 {i+1}: 蚂蚁数量={len(state.ants)}, P0 HP={state.bases[0].hp}, P1 HP={state.bases[1].hp}")
    if state.ants:
        print(f"  第一只蚂蚁: {state.ants[0]}")

print("\n=== 详细检查一只蚂蚁 ===")
if state.ants:
    ant = state.ants[0]
    print(f"蚂蚁类型: {type(ant)}")
    print(f"蚂蚁属性: {[a for a in dir(ant) if not a.startswith('_')]}")
    print(f"蚂蚁HP: {ant.hp if hasattr(ant, 'hp') else 'N/A'}")
    print(f"蚂蚁位置: {ant.x if hasattr(ant, 'x') else 'N/A'}, {ant.y if hasattr(ant, 'y') else 'N/A'}")
    print(f"蚂蚁所属: {ant.player if hasattr(ant, 'player') else 'N/A'}")