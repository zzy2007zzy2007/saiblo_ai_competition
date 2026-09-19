from typing import Any, Dict, List


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合多场对战结果

    语义与 ppo_v1 保持一致：
    - agent1_wins / agent2_wins / draws 记录胜负
    - agent1_first_move_* 记录 agent1 先手时的统计
    - agent1_second_move_* 记录 agent1 后手时的统计

    Returns:
        包含 total_battles、completed_battles、error_battles、
        agent1_wins/agent2_wins/draws、
        agent1_first_move_wins、agent1_first_move_battles、
        agent1_second_move_wins、agent1_second_move_battles、
        total_duration、avg_rounds 等统计指标
    """
    stats = {
        "total_battles": len(results),
        "completed_battles": 0,
        "error_battles": 0,
        "agent1_wins": 0,
        "agent2_wins": 0,
        "draws": 0,
        "agent1_first_move_wins": 0,
        "agent1_first_move_battles": 0,
        "agent1_second_move_wins": 0,
        "agent1_second_move_battles": 0,
        "avg_rounds": 0.0,
        "total_duration": 0.0,
    }

    total_rounds = 0
    for r in results:
        if "error" in r:
            stats["error_battles"] += 1
            continue

        stats["completed_battles"] += 1
        stats["total_duration"] += r.get("duration", 0)
        total_rounds += r.get("total_rounds", 0)

        result = r.get("result")
        first_player = r.get("first_player", 0)

        if result == "agent1_win":
            stats["agent1_wins"] += 1
            if first_player == 0:
                stats["agent1_first_move_wins"] += 1
            else:
                stats["agent1_second_move_wins"] += 1
        elif result == "agent2_win":
            stats["agent2_wins"] += 1
        else:
            stats["draws"] += 1

        if first_player == 0:
            stats["agent1_first_move_battles"] += 1
        else:
            stats["agent1_second_move_battles"] += 1

    if stats["completed_battles"] > 0:
        stats["avg_rounds"] = total_rounds / stats["completed_battles"]

    return stats
