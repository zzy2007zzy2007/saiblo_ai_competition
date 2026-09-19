from typing import Dict, List, Any


def aggregate_results(results: List[Dict]) -> Dict:
    aggregated = {
        'total_battles': len(results),
        'completed_battles': 0,
        'error_battles': 0,
        'agent1_wins': 0,
        'agent2_wins': 0,
        'draws': 0,
        'agent1_first_move_wins': 0,
        'agent1_first_move_battles': 0,
        'agent1_second_move_wins': 0,
        'agent1_second_move_battles': 0,
        'total_duration': 0.0,
        'avg_rounds': 0.0,
        'start_time': None,
        'end_time': None
    }

    if not results:
        return aggregated

    aggregated['start_time'] = min(r['start_time'] for r in results)
    aggregated['end_time'] = max(r['end_time'] for r in results)

    total_rounds = 0

    for result in results:
        aggregated['total_duration'] += result['duration']

        if result['result'] == 'error':
            aggregated['error_battles'] += 1
        else:
            aggregated['completed_battles'] += 1
            total_rounds += result.get('total_rounds', 0)

            if result['result'] == 'agent1_win':
                aggregated['agent1_wins'] += 1
            elif result['result'] == 'agent2_win':
                aggregated['agent2_wins'] += 1
            else:
                aggregated['draws'] += 1

            if result['first_player'] == 0:
                aggregated['agent1_first_move_battles'] += 1
                if result['result'] == 'agent1_win':
                    aggregated['agent1_first_move_wins'] += 1
            else:
                aggregated['agent1_second_move_battles'] += 1
                if result['result'] == 'agent1_win':
                    aggregated['agent1_second_move_wins'] += 1

    if aggregated['completed_battles'] > 0:
        aggregated['avg_rounds'] = total_rounds / aggregated['completed_battles']

    return aggregated
