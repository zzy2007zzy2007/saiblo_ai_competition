#!/usr/bin/env python3
"""检查 state 对象的属性"""

import sys
sys.path.insert(0, '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python')
sys.path.insert(0, '/root/autodl-tmp/AntWar/baselines/ann_593')

from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend

backend = load_backend(prefer_native=False)
game_state = backend.initial_state(seed=0)
state = PythonBackendState(game_state)

print('State attributes:')
for attr in dir(state):
    if not attr.startswith('_'):
        print(f'  {attr}')

print('\nTowers structure:')
print(f'  type: {type(state.towers)}')
if state.towers:
    print(f'  first tower type: {type(state.towers[0])}')
    print(f'  first tower attrs: {[a for a in dir(state.towers[0]) if not a.startswith("_")]}')