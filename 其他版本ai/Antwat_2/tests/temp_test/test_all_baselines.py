#!/usr/bin/env python3
"""Test all baseline models with their own SDKs - Working Version"""
import os
import sys
import time
from datetime import datetime

BASE_DIR = '/root/autodl-tmp/AntWar'

original_sys_path = sys.path.copy()

# First load main SDK for battle simulator
main_sdk_base = os.path.join(BASE_DIR, 'baselines', 'ann_593')
main_sdk_path = os.path.join(main_sdk_base, 'SDK')
if os.path.exists(main_sdk_base):
    sys.path.insert(0, main_sdk_base)
if os.path.exists(main_sdk_path):
    sys.path.insert(0, main_sdk_path)

# Now we can import battle components
from SDK.backend.state import PythonBackendState
from SDK.backend.core import load_backend
from SDK.backend.model import Operation as BackendOperation
from SDK.utils.constants import OperationType as BackendOperationType, MAX_ROUND
from antwar.coord import Coord
from antwar.protocol import build_tower_op


def print_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'[{timestamp}] {message}')


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


def load_agent(model_name):
    """Load a baseline agent with its own SDK"""
    model_path = os.path.join(BASE_DIR, 'baselines', model_name)
    sdk_path = os.path.join(model_path, 'SDK')

    if not os.path.exists(model_path):
        print_log(f'✗ Model path not found: {model_name}')
        return None

    temp_path = sys.path.copy()
    try:
        sys.path = []
        if os.path.exists(sdk_path):
            sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)

        for p in original_sys_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)

        ai_file = os.path.join(model_path, 'ai.py')
        if os.path.exists(ai_file):
            from ai import AI
            print_log(f'✓ Loaded {model_name} AI')
            sys.path = temp_path
            return AI, model_path, sdk_path
        else:
            print_log(f'✗ ai.py not found: {model_name}')
            sys.path = temp_path
            return None
    except Exception as e:
        import traceback
        print_log(f'✗ Load failed: {model_name}: {e}')
        print_log(traceback.format_exc()[:500])
        sys.path = temp_path
        return None


def get_agent_ops(agent, state, player, model_path, sdk_path):
    """Get agent operations with its SDK"""
    temp_path = sys.path.copy()
    try:
        sys.path = []
        if sdk_path:
            sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)
        for p in original_sys_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)
        
        if hasattr(agent, 'choose_bundle') and hasattr(agent, 'list_bundles'):
            bundles = agent.list_bundles(state, player)
            if bundles:
                bundle = agent.choose_bundle(state, player, bundles)
                return list(bundle.operations)
        
        if hasattr(agent, 'choose_operations'):
            return agent.choose_operations(state, player)
        
        return []
    except Exception as e:
        import traceback
        print_log(f'✗ Get ops failed: {e}')
        sys.path = temp_path
        return []
    finally:
        sys.path = temp_path


def run_single_battle(agent, model_name, model_path, sdk_path, opponent, episode, max_rounds=100):
    backend = load_backend(prefer_native=False)
    game_state = backend.initial_state(seed=episode)
    state = PythonBackendState(game_state)

    round_count = 0
    start_time = time.time()

    while not state.terminal and round_count < max_rounds:
        round_count += 1

        ops_agent = get_agent_ops(agent, state, 0, model_path, sdk_path)
        ops_opponent = opponent.choose_operations(state, 1)

        state.resolve_turn(ops_agent, ops_opponent)

        if round_count % 20 == 0:
            print_log(f'Round {round_count}: HP={state.bases[0].hp} vs {state.bases[1].hp}')

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


def test_model(model_name, episodes=2):
    print_log(f'\n{"=" * 70}')
    print_log(f'Testing model: {model_name}')
    print_log(f'{"=" * 70}')

    agent_info = load_agent(model_name)
    if agent_info is None:
        return None

    agent_class, model_path, sdk_path = agent_info
    agent = agent_class()

    opponent = BasicTowerAI()

    results = {
        'wins': 0, 'losses': 0, 'draws': 0,
        'total_rounds': 0, 'total_duration': 0
    }

    for episode in range(episodes):
        print_log(f'\n--- Episode {episode + 1} ---')
        battle_result = run_single_battle(agent, model_name, model_path, sdk_path, opponent, episode)

        results['total_rounds'] += battle_result['rounds']
        results['total_duration'] += battle_result['duration']

        if battle_result['result'] == 'win':
            results['wins'] += 1
            print_log(f'✓ Episode {episode + 1}: WIN')
        elif battle_result['result'] == 'loss':
            results['losses'] += 1
            print_log(f'✗ Episode {episode + 1}: LOSS')
        else:
            results['draws'] += 1
            print_log(f'— Episode {episode + 1}: DRAW')

        print_log(f'  Final HP: {battle_result["final_hp"][0]} vs {battle_result["final_hp"][1]}')
        print_log(f'  Rounds: {battle_result["rounds"]}, Duration: {battle_result["duration"]:.2f}s')

    print_log(f'\n{"=" * 70}')
    print_log(f'{model_name} Summary')
    print_log(f'{"=" * 70}')
    win_rate = results['wins'] / episodes * 100 if episodes > 0 else 0
    print_log(f'Wins: {results["wins"]}, Losses: {results["losses"]}, Draws: {results["draws"]}')
    print_log(f'Win Rate: {win_rate:.1f}%')
    print_log(f'Total Rounds: {results["total_rounds"]}, Total Duration: {results["total_duration"]:.2f}s')

    return results


def main():
    print_log('=' * 70)
    print_log('Baseline Battle Test - All Models')
    print_log('=' * 70)

    models = ['ann_593', 'gen99', 'gen199']
    all_results = {}

    for model_name in models:
        result = test_model(model_name, episodes=2)
        if result:
            all_results[model_name] = result

    print_log(f'\n{"=" * 70}')
    print_log('Final Summary - All Models')
    print_log(f'{"=" * 70}')

    for model_name, result in all_results.items():
        win_rate = result['wins'] / 2 * 100 if 2 > 0 else 0
        print_log(f'{model_name}: {result["wins"]}W/{result["losses"]}L/{result["draws"]}D, Win Rate: {win_rate:.1f}%')


if __name__ == '__main__':
    main()

