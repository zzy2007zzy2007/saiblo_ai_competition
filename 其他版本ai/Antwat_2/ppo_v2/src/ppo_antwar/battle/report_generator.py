class BattleReporter:
    """对战报告生成器 - 生成可读的对战评估报告"""

    def generate_report(self, results: dict, agent1_name: str, agent2_name: str) -> str:
        """生成对战报告字符串"""
        wins = results.get("agent1_wins", 0)
        losses = results.get("agent2_wins", 0)
        draws = results.get("draws", 0)
        total = wins + losses + draws
        win_rate = wins / max(total, 1)

        lines = []
        lines.append(f"=== Battle Report: {agent1_name} vs {agent2_name} ===")
        lines.append(f"Total battles: {results.get('total_battles', 0)}")
        lines.append(f"Completed: {results.get('completed_battles', 0)}")
        lines.append(f"Errors: {results.get('error_battles', 0)}")
        lines.append(f"Wins: {wins}")
        lines.append(f"Losses: {losses}")
        lines.append(f"Draws: {draws}")
        lines.append(f"Win rate: {win_rate:.2%}")
        lines.append(f"Avg rounds: {results.get('avg_rounds', 0):.1f}")
        w1 = results.get("agent1_first_move_wins", 0)
        b1 = results.get("agent1_first_move_battles", 1)
        lines.append(f"First move win rate: {w1}/{b1}")
        if results.get("agent1_second_move_battles", 0) > 0:
            w2 = results.get("agent1_second_move_wins", 0)
            b2 = results.get("agent1_second_move_battles", 0)
            lines.append(f"Second move win rate: {w2}/{b2}")
        lines.append(f"Total duration: {results.get('total_duration', 0):.1f}s")
        return "\n".join(lines)
