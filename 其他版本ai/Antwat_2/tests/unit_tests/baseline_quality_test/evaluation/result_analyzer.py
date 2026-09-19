"""结果分析模块"""
import json
import os

class ResultAnalyzer:
    """结果分析器类"""
    
    @staticmethod
    def analyze_results(results):
        """分析对战结果
        
        Args:
            results: 对战结果列表
            
        Returns:
            dict: 分析结果
        """
        analysis = {
            "total_games": 0,
            "total_wins": {},
            "total_draws": 0,
            "win_rates": {},
            "draw_rates": {},
            "strategy_performance": {},
            "head_to_head": {}
        }
        
        # 统计总游戏数和各策略胜场
        for result in results:
            agent1 = result["agent1"]
            agent2 = result["agent2"]
            wins1 = result["wins1"]
            wins2 = result["wins2"]
            draws = result.get("draws", 0)
            episodes = result.get("episodes", wins1 + wins2 + draws)
            
            # 确保episodes不为零
            if episodes == 0:
                continue
                
            analysis["total_games"] += episodes
            analysis["total_draws"] += draws
            
            # 统计各策略胜场
            if agent1 not in analysis["total_wins"]:
                analysis["total_wins"][agent1] = 0
            analysis["total_wins"][agent1] += wins1
            
            if agent2 not in analysis["total_wins"]:
                analysis["total_wins"][agent2] = 0
            analysis["total_wins"][agent2] += wins2
            
            # 记录头对头对战结果
            key = f"{agent1}_vs_{agent2}"
            analysis["head_to_head"][key] = {
                "wins1": wins1,
                "wins2": wins2,
                "draws": draws,
                "win_rate1": wins1 / episodes * 100,
                "win_rate2": wins2 / episodes * 100,
                "draw_rate": draws / episodes * 100,
                "avg_rounds": result.get("avg_rounds", 0)
            }
        
        # 计算各策略胜率
        for strategy, wins in analysis["total_wins"].items():
            analysis["win_rates"][strategy] = wins / analysis["total_games"] * 100
        
        # 计算平局率
        analysis["draw_rates"] = analysis["total_draws"] / analysis["total_games"] * 100
        
        # 计算各策略性能排名
        sorted_strategies = sorted(
            analysis["win_rates"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        analysis["strategy_performance"] = [
            {
                "strategy": strategy,
                "win_rate": win_rate,
                "wins": analysis["total_wins"][strategy],
                "rank": i + 1
            }
            for i, (strategy, win_rate) in enumerate(sorted_strategies)
        ]
        
        return analysis
    
    @staticmethod
    def generate_report(results, output_file=None):
        """生成对战报告
        
        Args:
            results: 对战结果列表
            output_file: 输出文件路径，None则仅打印
        """
        analysis = ResultAnalyzer.analyze_results(results)
        
        # 生成报告文本
        report = []
        report.append("=" * 80)
        report.append("策略对战分析报告")
        report.append("=" * 80)
        report.append(f"总对战场次: {analysis['total_games']}")
        report.append(f"总平局场次: {analysis['total_draws']}")
        report.append(f"总体平局率: {analysis['draw_rates']:.2f}%")
        report.append("")
        
        # 策略性能排名
        report.append("策略性能排名:")
        report.append("-" * 80)
        for item in analysis["strategy_performance"]:
            report.append(f"{item['rank']}. {item['strategy']:10} | 胜率: {item['win_rate']:.2f}% | 胜场: {item['wins']}")
        report.append("")
        
        # 头对头对战结果
        report.append("头对头对战结果:")
        report.append("-" * 80)
        for key, data in analysis["head_to_head"].items():
            agent1, agent2 = key.split("_vs_")
            report.append(f"{agent1:10} vs {agent2:10}")
            report.append(f"  {agent1}: {data['win_rate1']:.2f}% ({data['wins1']}胜)")
            report.append(f"  {agent2}: {data['win_rate2']:.2f}% ({data['wins2']}胜)")
            report.append(f"  平局: {data['draw_rate']:.2f}% ({data['draws']}场)")
            report.append(f"  平均回合数: {data['avg_rounds']:.2f}")
            report.append("")
        
        report.append("=" * 80)
        report_text = "\n".join(report)
        
        # 打印报告
        print(report_text)
        
        # 保存报告
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"\n报告已保存到: {output_file}")
        
        return analysis
    
    @staticmethod
    def export_results(results, output_file):
        """导出结果为JSON文件
        
        Args:
            results: 对战结果列表
            output_file: 输出文件路径
        """
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"结果已导出到: {output_file}")
