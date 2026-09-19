#!/usr/bin/env python3
"""Test ann_593 with its own SDK"""
import os
import sys

# Add main SDK path
main_sdk_path = '/root/autodl-tmp/AntWar/saiblo-antwar-sdk-python'
if os.path.exists(main_sdk_path):
    sys.path.insert(0, main_sdk_path)

# Load ann_593 with its own SDK
model_path = '/root/autodl-tmp/AntWar/baselines/ann_593'
sdk_path = os.path.join(model_path, 'SDK')

original_path = sys.path.copy()

try:
    # Add ann_593's paths first
    sys.path = []
    if os.path.exists(sdk_path):
        sys.path.insert(0, sdk_path)
    sys.path.insert(0, model_path)
    
    # Add back necessary paths
    for p in original_path:
        if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
            sys.path.append(p)
    
    from ai import AI
    print('✓ Loaded ann_593 AI')
    
    # Try to use FeatureExtractor
    from SDK.utils.features import FeatureExtractor
    print('✓ Loaded FeatureExtractor')
    
    # Now restore path and use main SDK for battle
    sys.path = original_path.copy()
    
    from SDK.backend.state import PythonBackendState
    from SDK.backend.core import load_backend
    from SDK.backend.model import Operation as BackendOperation
    from SDK.utils.constants import OperationType as BackendOperationType, MAX_ROUND
    from antwar.coord import Coord
    from antwar.protocol import build_tower_op
    import time
    
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
    
    print('\n=== Running battle test ===')
    
    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=0)
    state = PythonBackendState(game_state)
    
    print(f'Initial HP: Agent={state.bases[0].hp}, Opponent={state.bases[1].hp}')
    
    round_count = 0
    start_time = time.time()
    
    while not state.terminal and round_count < 100:
        round_count += 1
        
        # Get agent operations - use ann_593 SDK when needed
        current_agent = None
        try:
            temp_path = sys.path.copy()
            sys.path = []
            if os.path.exists(sdk_path):
                sys.path.insert(0, sdk_path)
            sys.path.insert(0, model_path)
            for p in temp_path:
                if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                    sys.path.append(p)
            
            if current_agent is None:
                from ai import AI as Agent
                current_agent = Agent()
            
            ops_agent = current_agent.choose_operations(state, 0)
            sys.path = temp_path
        except Exception as e:
            print(f'✗ Error getting agent operations: {e}')
            ops_agent = []
            sys.path = temp_path
        
        opponent = BasicTowerAI()
        ops_opponent = opponent.choose_operations(state, 1)
        
        state.resolve_turn(ops_agent, ops_opponent)
        
        if round_count % 20 == 0:
            print(f'Round {round_count}: HP Agent={state.bases[0].hp}, Opponent={state.bases[1].hp}')
    
    duration = time.time() - start_time
    result = 'draw'
    if state.winner == 0:
        result = 'win'
    elif state.winner == 1:
        result = 'loss'
    
    print(f'\nFinal: Result={result}, Rounds={round_count}, Duration={duration:.2f}s')
    print(f'Final HP: Agent={state.bases[0].hp}, Opponent={state.bases[1].hp}')
    
except Exception as e:
    import traceback
    print(f'✗ Error: {e}')
    print(traceback.format_exc())
    sys.path = original_path

