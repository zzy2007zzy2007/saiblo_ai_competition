import json
import os
from typing import Any, Dict, List

from loguru import logger


def save_round_logs(
    phase: str,
    generation: int,
    gen_dir: str,
    battle_results: List[Dict[str, Any]],
) -> str:
    """保存对战回合日志到 JSON 文件（统一格式）。

    所有阶段（pruning / touchstone / round_robin / baseline）均使用此函数
    持久化对局数据，确保字段名、文件结构完全一致。

    Args:
        phase: 阶段名称。
        generation: 代次。
        gen_dir: 代次输出目录。
        battle_results: _execute_battle() 返回的原始对战结果列表。

    Returns:
        保存的文件路径。
    """
    os.makedirs(gen_dir, exist_ok=True)
    filepath = os.path.join(gen_dir, f"{phase}_battles.json")

    battles = []
    for br in battle_results:
        if "error" in br:
            # 异常对局同样保留，仅记录错误信息
            battles.append({
                "home": br.get("home", ""),
                "away": br.get("away", ""),
                "result": "error",
                "error": br.get("error", "unknown"),
            })
            continue

        battle = {
            "home": br.get("home", ""),
            "away": br.get("away", ""),
            "result_code": br.get("result_code", 0),
            "result": br.get("result"),
            "first_player": br.get("first_player"),
            "total_rounds": br.get("total_rounds"),
            "illegal_home": br.get("illegal_actions_agent1", 0),
            "illegal_away": br.get("illegal_actions_agent2", 0),
        }
        # 保存回合详情（与 touchstone 格式一致）
        rounds_data = br.get("rounds", [])
        if rounds_data:
            battle["rounds"] = [
                {
                    "round": rd.get("round"),
                    "a1_ops": rd.get("agent1_ops"),
                    "a2_ops": rd.get("agent2_ops"),
                    "a1_coins": rd.get("agent1_coins_before"),
                    "a2_coins": rd.get("agent2_coins_before"),
                    "a1_hp": rd.get("agent1_hp_before"),
                    "a2_hp": rd.get("agent2_hp_before"),
                    "a1_towers": rd.get("agent1_towers"),
                    "a2_towers": rd.get("agent2_towers"),
                    "a1_hp_dmg": rd.get("agent1_hp_dmg"),
                    "a2_hp_dmg": rd.get("agent2_hp_dmg"),
                    "a1_ops_cost": rd.get("agent1_ops_cost"),
                    "a2_ops_cost": rd.get("agent2_ops_cost"),
                    "a1_illegal": rd.get("agent1_illegal"),
                    "a2_illegal": rd.get("agent2_illegal"),
                }
                for rd in rounds_data
            ]
        battles.append(battle)

    with open(filepath, "w") as f:
        json.dump({
            "phase": phase,
            "generation": generation,
            "total_battles": len(battles),
            "battles_with_rounds": sum(1 for b in battles if "rounds" in b),
            "battles": battles,
        }, f, indent=2, default=str)

    logger.info(
        f"[phase={phase}][gen={generation}] battles saved: "
        f"{len(battles)} total, "
        f"{sum(1 for b in battles if 'rounds' in b)} with rounds "
        f"→ {filepath}"
    )
    return filepath


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


def aggregate_illegal_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """从对战结果中聚合非法动作统计。

    Args:
        results: battle_simulator._execute_battle() 返回的对战结果列表，
                 每个结果含 illegal_actions_agent1 / illegal_actions_agent2。

    Returns:
        {total_illegal, agent1_illegal, agent2_illegal, num_battles, avg_rounds}
    """
    total_a1 = 0
    total_a2 = 0
    total_rounds = 0
    num_battles = 0

    for r in results:
        if "error" in r:
            continue
        ia1 = r.get("illegal_actions_agent1", 0)
        ia2 = r.get("illegal_actions_agent2", 0)
        total_a1 += ia1
        total_a2 += ia2
        total_rounds += r.get("total_rounds", 0)
        num_battles += 1

    return {
        "total_illegal": total_a1 + total_a2,
        "agent1_illegal": total_a1,
        "agent2_illegal": total_a2,
        "num_battles": num_battles,
        "avg_rounds": round(total_rounds / max(num_battles, 1), 1),
    }
