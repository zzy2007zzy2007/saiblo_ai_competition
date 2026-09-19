import time
from typing import Dict


def generate_report(aggregated: Dict, agent1_name: str, agent2_name: str) -> str:
    report = []
    report.append("=" * 60)
    report.append("        Baseline 对战评测报告")
    report.append("=" * 60)
    report.append("")
    report.append(f"执行时间: {aggregated.get('start_time', 'N/A')}")
    report.append(f"结束时间: {aggregated.get('end_time', 'N/A')}")
    report.append(f"对战双方: {agent1_name} vs {agent2_name}")
    report.append("=" * 60)

    report.append("")
    report.append("【总体统计】")
    report.append(f"总对战数: {aggregated['total_battles']}")
    report.append(f"完成对战: {aggregated['completed_battles']}")
    report.append(f"异常对战: {aggregated['error_battles']}")
    report.append(f"总耗时: {aggregated['total_duration']:.2f}秒")
    report.append(f"平均回合数: {aggregated.get('avg_rounds', 0.0):.1f}")
    report.append("=" * 60)

    report.append("")
    report.append("【对战结果】")
    report.append("┌────────────┬────────────┬───────┬───────┬──────┬────────┐")
    report.append("│   Agent1   │   Agent2   │  胜   │  负  │  平   │  胜率   │")
    report.append("├────────────┼────────────┼───────┼──────┼───────┼────────┤")

    total = aggregated['agent1_wins'] + aggregated['agent2_wins'] + aggregated['draws']
    win_rate = (aggregated['agent1_wins'] / total * 100) if total > 0 else 0.0
    report.append(f"│ {agent1_name:^10} │ {agent2_name:^10} │ {aggregated['agent1_wins']:^5} │ {aggregated['agent2_wins']:^4} │ {aggregated['draws']:^5} │ {win_rate:^8.1f}% │")

    report.append("└────────────┴────────────┴───────┴──────┴───────┴────────┘")

    report.append("")
    report.append("【详细对战记录】")
    report.append(f"{agent1_name} vs {agent2_name}: {aggregated['agent1_wins']}胜 {aggregated['agent2_wins']}负 {aggregated['draws']}平 (胜率 {win_rate:.1f}%)")

    report.append("")
    report.append("【先后手胜率】")
    first_battles = aggregated['agent1_first_move_battles']
    first_wins = aggregated['agent1_first_move_wins']
    first_win_rate = (first_wins / first_battles * 100) if first_battles > 0 else 0.0
    first_losses = first_battles - first_wins

    second_battles = aggregated['agent1_second_move_battles']
    second_wins = aggregated['agent1_second_move_wins']
    second_win_rate = (second_wins / second_battles * 100) if second_battles > 0 else 0.0
    second_losses = second_battles - second_wins

    report.append(f"{agent1_name} - 先手: {first_wins}胜 {first_losses}负 ({first_win_rate:.1f}%)")
    report.append(f"{agent1_name} - 后手: {second_wins}胜 {second_losses}负 ({second_win_rate:.1f}%)")

    report.append("")
    report.append("=" * 60)
    report.append(f"报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)

    return "\n".join(report)
