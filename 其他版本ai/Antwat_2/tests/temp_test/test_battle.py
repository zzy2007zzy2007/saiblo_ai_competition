#!/usr/bin/env python3
"""Simple test script for baselines - Working Version"""
import os
import sys
import time
from datetime import datetime

BASE_DIR = '/root/autodl-tmp/AntWar'
original_sys_path = sys.path.copy()


def print_log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print(f'[{timestamp}] {message}')


def test_model(model_name):
    model_path = os.path.join(BASE_DIR, 'baselines', model_name)
    sdk_path = os.path.join(model_path, 'SDK')

    if not os.path.exists(model_path):
        print_log(f'✗ Model path not found: {model_name}')
        return None

    # Load this model with its own SDK
    temp_path = sys.path.copy()
    try:
        sys.path = []
        if os.path.exists(sdk_path):
            sys.path.insert(0, sdk_path)
        sys.path.insert(0, model_path)

        for p in original_sys_path:
            if 'site-packages' in p or 'python3.12' in p or 'lib/python' in p:
                sys.path.append(p)

        from ai import AI
        print_log(f'✓ Loaded {model_name} AI')
        agent = AI()

        from SDK.backend.state import PythonBackendState
        from SDK.backend.core import load_backend
        from SDK.backend.model import Operation as BackendOperation
        from SDK.utils.constants import OperationType as BackendOperationType, MAX_ROUND
        from SDK.utils.actions import ActionBundle
        from antwar.coord import Coord
        from antwar.protocol import build_tower_op

        # Create basic tower opponent (uses same SDK for battle)
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

        opponent = BasicTowerAI()

        results = {
            'wins': 0, 'losses': 0, 'draws': 0,
            'total_rounds': 0, 'total_duration': 0
        }

        for episode in range(2):
            print_log(f'\n--- {model_name} - Episode {episode + 1} ---')

            backend = load_backend(prefer_native=False)
            game_state = backend.initial_state(seed=episode)
            state = PythonBackendState(game_state)

            round_count = 0
            start_time = time.time()

            while not state.terminal and round_count < 100:
                round_count += 1

                if hasattr(agent, 'choose_bundle') and hasattr(agent, 'list_bundles'):
                    bundles = agent.list_bundles(state, 0)
                    if bundles:
                        bundle = agent.choose_bundle(state, 0, bundles)
                        ops_agent = list(bundle.operations)
                    else:
                        ops_agent = []
                elif hasattr(agent, 'choose_operations'):
                    ops_agent = agent.choose_operations(state, 0)
                else:
                    ops_agent = []

                ops_opponent = opponent.choose_operations(state, 1)
                state.resolve_turn(ops_agent, ops_opponent)

                if round_count % 20 == 0:
                    print_log(f'Round {round_count}: HP={state.bases[0].hp} vs {state.bases[1].hp}')

            duration = time.time() - start_time
            result = 'draw'
            if state.winner == 0:
                result = 'win'
                results['wins'] += 1
            elif state.winner == 1:
                result = 'loss'
                results['losses'] += 1
            else:
                results['draws'] += 1

            results['total_rounds'] += round_count
            results['total_duration'] += duration

            if result == 'win':
                print_log(f'✓ Episode {episode + 1}: WIN')
            elif result == 'loss':
                print_log(f'✗ Episode {episode + 1}: LOSS')
            else:
                print_log(f'— Episode {episode + 1}: DRAW')

            print_log(f'  Final HP: {state.bases[0].hp} vs {state.bases[1].hp}')
            print_log(f'  Rounds: {round_count}, Duration: {duration:.2f}s')

        win_rate = results['wins'] / 2 * 100 if 2 > 0 else 0
        print_log(f'\n{"=" * 70}')
        print_log(f'{model_name} Summary')
        print_log(f'{"=" * 70}')
        print_log(f'Wins: {results["wins"]}, Losses: {results["losses"]}, Draws: {results["draws"]}')
        print_log(f'Win Rate: {win_rate:.1f}%')
        print_log(f'Total Rounds: {results["total_rounds"]}, Total Duration: {results["total_duration"]:.2f}s')

        sys.path = temp_path
        return results

    except Exception as e:
        import traceback
        print_log(f'✗ Test failed: {model_name}: {e}')
        print_log(traceback.format_exc())
        sys.path = temp_path
        return None


def main():
    print_log('=' * 70)
    print_log('Baseline Battle Test - All Models')
    print_log('=' * 70)

    models = ['ann_593', 'gen99', 'gen199']
    all_results = {}

    for model_name in models:
        result = test_model(model_name)
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

