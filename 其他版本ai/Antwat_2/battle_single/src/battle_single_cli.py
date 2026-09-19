#!/usr/bin/env python3
"""
Battle Single 简版对战系统 - CLI 入口

================================================================================
WARNING: 不要在本地测试！请在 Server5 的 /tmp/battle 文件夹测试！
WARNING: Do NOT test locally! Please test in /tmp/battle on Server5!
================================================================================

用法:
    python -m battle_single.src.battle_single_cli [选项]

示例:
    python -m battle_single.src.battle_single_cli --agent1 ann_v1 --agent2 gen99
    python -m battle_single.src.battle_single_cli --config /tmp/battle_single/config.yaml
"""

import argparse
import json
import os
import sys
import time

from .battle_single_logger import BattleLogger, LogLevel
from .battle_single_config import BattleSingleConfig
from .agent_loader import AgentLoader


def aggregate_results(results):
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

            if result['first_player'] == result['agent1_name']:
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


def generate_report(aggregated, agent1_name, agent2_name):
    report = []
    report.append("=" * 60)
    report.append("        Battle Single 对战评测报告")
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


def save_results(results, output_result_json):
    output_dir = os.path.dirname(output_result_json) or '/tmp/battle_single'
    os.makedirs(output_dir, exist_ok=True)

    with open(output_result_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Battle Single 简版对战系统")
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--episodes', type=int, default=None, help='对战局数（每局交换先后手）')
    parser.add_argument('--workers', type=int, default=None, help='并行线程数')
    parser.add_argument('--max-rounds', type=int, default=None, help='单场对战最大回合数')
    parser.add_argument('--log-level', type=str, default=None, help='日志级别: DEBUG, INFO, WARNING, ERROR')
    parser.add_argument('--enable-timing', action='store_true', default=None, help='启用性能计时')
    parser.add_argument('--disable-timing', action='store_true', default=None, help='禁用性能计时')
    parser.add_argument('--agents', type=str, nargs='+', default=None, help='指定对战的两个 agent')
    parser.add_argument('--agent1', type=str, default=None, help='第一个 agent 名称')
    parser.add_argument('--agent2', type=str, default=None, help='第二个 agent 名称')
    parser.add_argument('--output', type=str, default=None, help='报告输出文件路径')
    parser.add_argument('--no-console-log', action='store_true', help='不在控制台输出日志')

    args = parser.parse_args()

    config = BattleSingleConfig()

    if args.config:
        config.load_from_file(args.config)

    config.load_from_args(args)

    if args.enable_timing is not None:
        config.logging_enable_timing = args.enable_timing
    if args.disable_timing is not None:
        config.logging_enable_timing = not args.disable_timing

    if not config.agent1_name or not config.agent2_name:
        print("错误: 必须指定 --agent1 和 --agent2 参数，或使用 --agents 参数")
        sys.exit(1)

    log_level = config.get_log_level()
    log_to_console = not args.no_console_log

    logger = BattleLogger(
        agent1_name=config.agent1_name,
        agent2_name=config.agent2_name,
        log_level=log_level,
        log_to_console=log_to_console,
        enable_timing=config.logging_enable_timing
    )

    logger.info('start', "=" * 60)
    logger.info('start', "    Battle Single 简版对战系统")
    logger.info('start', "=" * 60)
    logger.info('config', f"对战双方: {config.agent1_name} vs {config.agent2_name}")
    logger.info('config', f"对战局数: {config.battle_num_episodes} (每局交换先后手)")
    logger.info('config', f"最大回合数: {config.battle_max_rounds}")
    logger.info('config', f"并行线程数: {config.battle_parallel_workers}")
    logger.info('config', f"日志级别: {log_level.name}")
    logger.info('config', f"日志目录: {logger.get_log_dir()}")
    logger.info('start', "=" * 60)

    agent_loader = AgentLoader(logger=logger)

    logger.info('load', "\n=== 加载 Agent ===")

    agent_load_start = time.time()
    agent1 = agent_loader.load_agent(config.agent1_name)
    logger.timing_start('load_agent1')
    if not agent1:
        logger.error('load', f"无法加载 agent1: {config.agent1_name}")
        sys.exit(1)
    logger.timing_stop('load_agent1')

    agent2 = agent_loader.load_agent(config.agent2_name)
    logger.timing_start('load_agent2')
    if not agent2:
        logger.error('load', f"无法加载 agent2: {config.agent2_name}")
        sys.exit(1)
    logger.timing_stop('load_agent2')
    agent_load_duration = time.time() - agent_load_start
    logger.info('load', f"Agent 加载完成，耗时: {agent_load_duration:.2f}秒")

    from .battle_single_simulator import BattleSingleSimulator

    simulator = BattleSingleSimulator(logger=logger, max_rounds=config.battle_max_rounds)

    logger.info('battle', "\n=== 开始对战评测 ===")
    start_time = time.time()

    results = simulator.run_battles_parallel(
        agent1, agent2,
        config.agent1_name, config.agent2_name,
        config.battle_num_episodes,
        parallel_workers=config.battle_parallel_workers
    )

    end_time = time.time()
    total_duration = end_time - start_time

    logger.info('battle', f"\n=== 对战评测完成 ===")
    logger.info('battle', f"总耗时: {total_duration:.2f}秒")

    aggregated = aggregate_results(results)
    report = generate_report(aggregated, config.agent1_name, config.agent2_name)
    
    if log_to_console:
        print("\n" + report)

    save_results(results, config.output_result_json)

    logger.info('end', f"\n✓ 评测结果已保存到: {config.output_result_json}")
    logger.info('end', f"✓ 详细日志已保存到: {logger.get_log_dir()}")
    
    logger.print_performance_report()
    logger.close()


if __name__ == "__main__":
    main()