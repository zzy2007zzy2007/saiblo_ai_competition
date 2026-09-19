"""策略比较测试脚本 - 使用官方SDK和baselines策略"""
import argparse
import json
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from tests.evaluation.official_battle_engine import OfficialBattleEngine
from tests.evaluation.baseline_strategy_loader import load_baseline_strategy, get_available_baseline_strategies

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="策略比较测试脚本 - 使用官方SDK")
    
    # 策略选择
    parser.add_argument(
        "--strategies", 
        nargs="+",
        default=["ann_593", "ann_v1", "sample", "gen199", "gen99"],
        help="要测试的策略列表"
    )
    
    # 对战轮数
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=20,
        help="每对策略的对战轮数"
    )
    
    # 最大回合数
    parser.add_argument(
        "--max-rounds", 
        type=int, 
        default=512,
        help="每局游戏的最大回合数（官方标准）"
    )
    
    # 结果导出
    parser.add_argument(
        "--export", 
        default="win_rates_official.json",
        help="导出结果到JSON文件"
    )
    
    # 报告输出
    parser.add_argument(
        "--report", 
        default="battle_report_official.txt",
        help="生成报告到指定文件"
    )
    
    # 详细输出
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="打印详细的对战信息"
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    print("=" * 80)
    print("策略对战分析 - 使用官方SDK和baselines策略")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"策略列表: {', '.join(args.strategies)}")
    print(f"对战轮数: {args.episodes}")
    print(f"最大回合数: {args.max_rounds}")
    print()
    
    # 加载策略
    print("=== 加载策略 ===")
    agents = {}
    for strategy_name in args.strategies:
        try:
            agent = load_baseline_strategy(strategy_name)
            agents[strategy_name] = agent
            print(f"✓ 成功加载策略: {strategy_name}")
        except Exception as e:
            print(f"✗ 加载策略 {strategy_name} 失败: {e}")
    
    if len(agents) < 2:
        print("错误: 至少需要2个策略才能进行比较")
        return
    
    print(f"\n成功加载 {len(agents)} 个策略")
    print()
    
    # 创建对战日志目录
    log_dir = os.path.join(os.path.dirname(__file__), "battle_logs_official")
    os.makedirs(log_dir, exist_ok=True)
    
    # 运行对战
    print("=== 开始对战 ===")
    results = []
    
    strategy_names = list(agents.keys())
    total_pairs = len(strategy_names) * (len(strategy_names) - 1)
    current_pair = 0
    
    for i, strategy1 in enumerate(strategy_names):
        for j, strategy2 in enumerate(strategy_names):
            if i == j:
                continue
            
            current_pair += 1
            print(f"\n[{current_pair}/{total_pairs}] {strategy1} vs {strategy2}")
            
            # 创建日志文件
            log_file = os.path.join(log_dir, f"{strategy1}_vs_{strategy2}.log")
            
            # 运行对战
            try:
                wins1, wins2, draws, avg_rounds = OfficialBattleEngine.run_battle(
                    agents[strategy1],
                    agents[strategy2],
                    episodes=args.episodes,
                    max_rounds=args.max_rounds,
                    verbose=args.verbose,
                    log_file=log_file
                )
                
                # 记录结果
                result = {
                    "agent1": strategy1,
                    "agent2": strategy2,
                    "wins1": wins1,
                    "wins2": wins2,
                    "draws": draws,
                    "win_rate1": wins1 / args.episodes * 100,
                    "win_rate2": wins2 / args.episodes * 100,
                    "draw_rate": draws / args.episodes * 100,
                    "avg_rounds": avg_rounds,
                    "episodes": args.episodes,
                    "log_file": f"battle_logs_official/{strategy1}_vs_{strategy2}.log"
                }
                results.append(result)
                
                print(f"  结果: {strategy1} {wins1}胜, {strategy2} {wins2}胜, {draws}平")
                print(f"  胜率: {strategy1} {result['win_rate1']:.1f}%, {strategy2} {result['win_rate2']:.1f}%")
                print(f"  平均回合数: {avg_rounds:.2f}")
                
            except Exception as e:
                print(f"  对战失败: {e}")
                import traceback
                traceback.print_exc()
    
    # 导出结果
    if args.export:
        export_path = os.path.join(os.path.dirname(__file__), args.export)
        with open(export_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已导出到: {export_path}")
    
    # 生成报告
    if args.report:
        report_path = os.path.join(os.path.dirname(__file__), args.report)
        generate_report(results, agents.keys(), report_path)
        print(f"报告已生成: {report_path}")
    
    print("\n" + "=" * 80)
    print("对战分析完成")
    print("=" * 80)

def generate_report(results, strategy_names, report_path):
    """生成对战报告"""
    # 统计每个策略的总胜场
    total_wins = {name: 0 for name in strategy_names}
    total_games = {name: 0 for name in strategy_names}
    
    for result in results:
        total_wins[result["agent1"]] += result["wins1"]
        total_games[result["agent1"]] += result["episodes"]
        total_wins[result["agent2"]] += result["wins2"]
        total_games[result["agent2"]] += result["episodes"]
    
    # 计算胜率
    win_rates = {name: total_wins[name] / total_games[name] * 100 if total_games[name] > 0 else 0 
                 for name in strategy_names}
    
    # 排序
    sorted_strategies = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
    
    # 生成报告
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("策略对战分析报告 - 使用官方SDK\n")
        f.write("=" * 80 + "\n")
        f.write(f"总对战场次: {sum(total_games.values()) // 2}\n")
        f.write(f"总平局场次: {sum(r['draws'] for r in results)}\n")
        f.write(f"总体平局率: {sum(r['draws'] for r in results) / sum(total_games.values()) * 100:.2f}%\n")
        f.write("\n")
        
        f.write("策略性能排名:\n")
        f.write("-" * 80 + "\n")
        for rank, (name, win_rate) in enumerate(sorted_strategies, 1):
            f.write(f"{rank}. {name:<12} | 胜率: {win_rate:5.2f}% | 胜场: {total_wins[name]}\n")
        
        f.write("\n")
        f.write("头对头对战结果:\n")
        f.write("-" * 80 + "\n")
        for result in results:
            f.write(f"\n{result['agent1']:<12} vs {result['agent2']:<12}\n")
            f.write(f"  {result['agent1']}: {result['win_rate1']:6.2f}% ({result['wins1']}胜)\n")
            f.write(f"  {result['agent2']}: {result['win_rate2']:6.2f}% ({result['wins2']}胜)\n")
            f.write(f"  平局: {result['draw_rate']:6.2f}% ({result['draws']}场)\n")
            f.write(f"  平均回合数: {result['avg_rounds']:.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n")

if __name__ == '__main__':
    main()
