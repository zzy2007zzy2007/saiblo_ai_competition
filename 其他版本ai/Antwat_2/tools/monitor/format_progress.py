#!/usr/bin/env python3
import json
import sys


def format_progress(data):
    lines = []

    status_val = data.get("status", "unknown")
    status_icon = "✅" if status_val == "success" else "⚠️"
    lines.append("采集状态: {} {}".format(status_icon, status_val))
    lines.append("采集时间: {}".format(data.get("timestamp", "N/A")))
    lines.append("训练目录: {}".format(data.get("training_dir", "N/A")))
    lines.append("")

    # Screen 会话状态
    lines.append("=" * 60)
    lines.append("   Screen 会话状态")
    lines.append("=" * 60)
    screen = data.get("screen", {})
    if screen.get("exists"):
        lines.append("状态: ✅ 训练会话存在 ({})".format(screen.get("session_name", "")))
        lines.append("会话ID: {}".format(screen.get("session_id", "N/A")))
        lines.append("创建时间: {}".format(screen.get("created_at", "N/A")))
        lines.append("状态: {}".format(screen.get("status", "N/A")))
    else:
        lines.append("状态: ❌ 训练会话不存在")
    lines.append("")

    # 训练进程状态
    lines.append("=" * 60)
    lines.append("   训练进程状态")
    lines.append("=" * 60)
    process = data.get("process")
    if process:
        lines.append("状态: ✅ 训练进程运行中")
        lines.append("PID: {}".format(process.get("pid", "N/A")))
        lines.append("CPU占用: {}%".format(process.get("cpu_percent", "N/A")))
        lines.append("内存占用: {} MB".format(process.get("memory_mb", "N/A")))
        lines.append("启动时间: {}".format(process.get("start_time", "N/A")))
    else:
        lines.append("状态: ❌ 训练进程未运行")
    lines.append("")

    # 训练指标（适配 ppo_v2 新结构）
    lines.append("=" * 60)
    lines.append("   训练指标状态")
    lines.append("=" * 60)
    tm = data.get("training_metrics", {})
    if "error" in tm:
        lines.append("状态: ⚠️ {}".format(tm.get("error", "未知错误")))
    elif tm.get("status") == "ok":
        lines.append("当前轮次: {}".format(tm.get("current_episode", 0)))
        lines.append("最新结果: {}".format(tm.get("last_result", "N/A")))
        lines.append("平均奖励: {:.4f}".format(tm.get("avg_reward", 0)))
        lines.append("平均回合数: {:.1f}".format(tm.get("avg_rounds", 0)))
        lines.append("策略损失: {:.6f}".format(tm.get("policy_loss", 0)))
        lines.append("价值损失: {:.6f}".format(tm.get("value_loss", 0)))
        lines.append("熵: {:.4f}".format(tm.get("entropy", 0)))
        lines.append("胜率: {:.2%}".format(tm.get("win_rate", 0)))
    else:
        lines.append("状态: ⚠️ 训练指标数据不完整")
        if tm.get("current_episode"):
            lines.append("当前轮次: {}".format(tm["current_episode"]))
        if tm.get("avg_reward"):
            lines.append("平均奖励: {:.4f}".format(tm["avg_reward"]))
    lines.append("")

    # 系统资源
    lines.append("=" * 60)
    lines.append("   系统资源采样")
    lines.append("=" * 60)
    sm = data.get("system_metrics", {})
    if "error" in sm:
        lines.append("状态: ⚠️ {}".format(sm.get("error", "未知错误")))
    else:
        lines.append("CPU 平均: {:.1f}% | 最大: {:.1f}%".format(sm.get("cpu_avg", 0), sm.get("cpu_max", 0)))
        lines.append("内存 平均: {:.1f}% | 最大: {:.1f}%".format(sm.get("ram_avg", 0), sm.get("ram_max", 0)))
        if sm.get("gpu_avg"):
            lines.append("GPU 平均: {:.1f}% | 最大: {:.1f}%".format(sm.get("gpu_avg", 0), sm.get("gpu_max", 0)))
        lines.append("样本数: {}".format(sm.get("sample_count", 0)))

        latest = sm.get("latest", {})
        if latest:
            lines.append("")
            lines.append("最新采样:")
            lines.append("  时间戳: {}".format(latest.get("timestamp", "N/A")))
            lines.append("  阶段: {}".format(latest.get("phase", "N/A")))
            lines.append("  CPU: {:.1f}%".format(latest.get("cpu_percent", 0)))
            lines.append("  内存: {:.1f}% ({:.1f}/{:.1f}GB)".format(
                latest.get("ram_percent", 0), latest.get("ram_used_gb", 0), latest.get("ram_total_gb", 0)))
            if latest.get("gpu_util") is not None:
                lines.append("  GPU利用率: {}%".format(latest.get("gpu_util", 0)))
                lines.append("  GPU显存: {}MB".format(latest.get("gpu_mem_used_mb", 0)))
                lines.append("  GPU温度: {}°C".format(latest.get("gpu_temp", 0)))
    lines.append("")

    # GPU 状态
    lines.append("=" * 60)
    lines.append("   GPU 状态")
    lines.append("=" * 60)
    gpu = data.get("gpu", {})
    if gpu:
        lines.append("型号: {}".format(gpu.get("name", "N/A")))
        lines.append("利用率: {}%".format(gpu.get("utilization", 0)))
        lines.append("显存: {}/{}MB".format(gpu.get("memory_used_mb", 0), gpu.get("memory_total_mb", 0)))
        lines.append("温度: {}°C".format(gpu.get("temperature", 0)))
    else:
        lines.append("状态: ⚠️ GPU信息不可用")
    lines.append("")

    # 对战统计（适配新结构：baseline winrates）
    lines.append("=" * 60)
    lines.append("   对战统计")
    lines.append("=" * 60)
    battle = data.get("battle_stats", {})
    if battle.get("exists"):
        bd = battle.get("data", {})
        for agent, stats in bd.items():
            lines.append("  {}: win_rate={:.1f}% ({}/{})".format(
                agent, stats.get("win_rate", 0) * 100,
                stats.get("wins", 0), stats.get("battles", 0)))
    else:
        lines.append("状态: ⚠️ {}".format(battle.get("error", "对战统计文件不存在")))

    ts = data.get("training_status", {})
    if ts:
        lines.append("")
        lines.append("训练状态详情:")
        if ts.get("win_rate") is not None:
            lines.append("  胜率: {:.2f}%".format(ts.get("win_rate", 0) * 100))
        if ts.get("total_wins") is not None:
            lines.append("  胜/负/平: {}/{}/{}".format(
                ts.get("total_wins", 0),
                ts.get("total_losses", 0),
                ts.get("total_draws", 0)
            ))
        if ts.get("best_avg_reward") is not None:
            lines.append("  最佳奖励: {:.4f}".format(ts.get("best_avg_reward", 0)))
        if ts.get("training_speed") is not None:
            lines.append("  训练速度: {:.2f} 轮/分钟".format(ts.get("training_speed", 0)))
        stability = ts.get("training_stability", {})
        if stability:
            lines.append("  奖励标准差: {:.4f}".format(stability.get("reward_std", 0)))
            lines.append("  损失标准差: {:.6f}".format(stability.get("loss_std", 0)))
    lines.append("")

    # 异常日志
    lines.append("=" * 60)
    lines.append("   异常记录检查")
    lines.append("=" * 60)
    err = data.get("error_log", {})
    if err.get("exists"):
        lines.append("异常记录数: {}".format(err.get("count", 0)))
    else:
        lines.append("状态: ✅ 无异常记录 (error.log 不存在)")
    lines.append("")

    lines.append("=" * 60)
    lines.append("   检查完成")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    json_file = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        if json_file:
            with open(json_file, "r") as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)

        print(format_progress(data))
    except Exception as e:
        print("解析错误: {}".format(str(e)), file=sys.stderr)
        sys.exit(1)
