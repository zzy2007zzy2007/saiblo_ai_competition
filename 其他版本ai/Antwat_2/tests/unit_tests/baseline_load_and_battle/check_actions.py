#!/usr/bin/env python3
"""检查规则型AI的候选动作"""

import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/gen99')

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from ai import AI as RuleBasedAI

backend = load_backend(prefer_native=False)
game_state = backend.initial_state(seed=0)
state = PythonBackendState(game_state)

agent = RuleBasedAI()

print("=== 检查 gen99 的候选动作 ===\n")

bundles = agent.list_bundles(state, 0)
print(f"总候选动作数: {len(bundles)}\n")

print("所有候选动作:")
for i, bundle in enumerate(bundles):
    ops_str = []
    for op in bundle.operations:
        ops_str.append(f"{op.op_type.name}({op.arg0},{op.arg1})")
    print(f"  [{i}] {bundle.name}: score={bundle.score:.2f}, ops={ops_str}")

print("\n=== 只选择有进攻的动作 ===")
attack_bundles = [b for b in bundles if any(op.op_type.name in ['BUILD_TOWER', 'UPGRADE_TOWER'] for op in b.operations)]
print(f"建造/升级塔的动作数: {len(attack_bundles)}")

move_bundles = [b for b in bundles if any(op.op_type.name.startswith('MOVE') for op in b.operations)]
print(f"移动蚂蚁的动作数: {len(move_bundles)}")

other_bundles = [b for b in bundles if b not in attack_bundles and b not in move_bundles]
print(f"其他动作数: {len(other_bundles)}")
for b in other_bundles[:5]:
    ops_str = [f"{op.op_type.name}" for op in b.operations]
    print(f"  - {b.name}: {ops_str}")