#!/bin/bash

# 状态管理函数 - 增强版
# 参考 ppo_v4 实现，适配本项目日志结构

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/python_env.sh"

# 检查训练是否正在运行
check_training_running() {
    local server_name="$1"
    local model_name="$2"
    local screen_session="${model_name}_training"

    local command="screen -ls | grep -q '${screen_session}'"
    execute_remote_command "$server_name" "$command"
    return $?
}

# 获取训练状态
get_training_status() {
    local server_name="$1"
    local model_name="$2"

    if check_training_running "$server_name" "$model_name"; then
        echo "running"
    else
        echo "stopped"
    fi
}

# 获取最新训练目录（优先使用时间戳目录，fallback到default）
get_latest_training_dir() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")

    local latest=$(execute_remote_command "$server_name" "ls -td ${MODEL_BASE_DIR}/outputs/20*/ 2>/dev/null | head -1")
    if [[ -n "$latest" ]]; then
        echo "$latest"
        return 0
    fi

    local latest=$(execute_remote_command "$server_name" "ls -td ${MODEL_BASE_DIR}/outputs/default/ 2>/dev/null | head -1")
    if [[ -n "$latest" ]]; then
        echo "$latest"
        return 0
    fi

    local latest=$(execute_remote_command "$server_name" "ls -td ${model_dir}/outputs/20*/ 2>/dev/null | head -1")
    if [[ -n "$latest" ]]; then
        echo "$latest"
        return 0
    fi

    echo ""
    return 1
}

# 获取训练日志
get_training_log() {
    local server_name="$1"
    local model_name="$2"
    
    # 尝试查找最新的训练日志
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")
    
    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi
    
    local log_dir="${latest_dir}training"
    local command="latest_log=\$(ls -t ${log_dir}/training_*.log 2>/dev/null | head -1); if [ -n \"\$latest_log\" ]; then tail -n 50 \"\$latest_log\"; else echo '日志文件不存在'; fi"
    execute_remote_command "$server_name" "$command"
}

# 获取训练统计信息
get_training_stats() {
    local server_name="$1"
    local model_name="$2"
    
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")
    
    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi
    
    # 检查训练统计文件
    local stats_file="$latest_dir/training/training_metrics.json"
    local command="if [ -f '$stats_file' ]; then cat '$stats_file'; else echo '{\"error\": \"统计文件不存在\"}'; fi"
    execute_remote_command "$server_name" "$command"
}

# 获取训练指标（从 batch_metrics.jsonl + selfplay_battle_log.jsonl 联合读取 — ppo_v2 适配）
get_training_metrics() {
    local server_name="$1"
    local model_name="$2"
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    echo "训练指标文件: ${latest_dir}training/"
    echo ""

    execute_remote_command "$server_name" "/root/miniconda3/bin/python -c '
import json

result = {}
try:
    # 从 batch_metrics.jsonl 读取最后一条
    with open(\"${latest_dir}training/batch_metrics.jsonl\") as f:
        for line in f:
            pass
        if line:
            batch = json.loads(line)
            result[\"episode\"] = batch.get(\"episode\", 0)
            result[\"policy_loss\"] = batch.get(\"policy_loss\", 0)
            result[\"value_loss\"] = batch.get(\"value_loss\", 0)
            result[\"entropy\"] = batch.get(\"entropy\", 0)
            result[\"grad_norm\"] = batch.get(\"grad_norm\", 0)
            result[\"clip_fraction\"] = batch.get(\"clip_fraction\", 0)
            result[\"learning_rate\"] = batch.get(\"learning_rate\", 0)
            result[\"reward_mean\"] = batch.get(\"reward_mean\", 0)
            result[\"return_mean\"] = batch.get(\"return_mean\", 0)

    # 从 selfplay_battle_log.jsonl 读取最后一条
    with open(\"${latest_dir}training/selfplay_battle_log.jsonl\") as f:
        for line in f:
            pass
        if line:
            battle = json.loads(line)
            result[\"reward\"] = battle.get(\"reward\", 0)
            result[\"rounds\"] = battle.get(\"rounds\", 0)
            result[\"result\"] = battle.get(\"result\", \"\")

    # 从 episode_batch_battle_stats.jsonl 读取最后的胜率
    with open(\"${latest_dir}training/episode_batch_battle_stats.jsonl\") as f:
        for line in f:
            pass
        if line:
            stats = json.loads(line)
            result[\"win_rate\"] = stats.get(\"win_rate\", 0)

    print(\"=\" * 70)
    print(\"[训练指标摘要] (batch_metrics 最后一条)\")
    print(\"=\" * 70)
    print(\"  当前轮次:     \" + str(result.get(\"episode\", \"N/A\")))
    print(\"  策略损失:     \" + str(round(result.get(\"policy_loss\", 0), 6)))
    print(\"  价值损失:     \" + str(round(result.get(\"value_loss\", 0), 6)))
    print(\"  熵:           \" + str(round(result.get(\"entropy\", 0), 4)))
    print(\"  梯度范数:     \" + str(round(result.get(\"grad_norm\", 0), 2)))
    print(\"  ClipFrac:     \" + str(round(result.get(\"clip_fraction\", 0), 4)))
    print(\"  学习率:       \" + str(result.get(\"learning_rate\", 0)))
    print(\"  平均奖励:     \" + str(round(result.get(\"reward_mean\", 0), 4)))
    print(\"  回报均值:     \" + str(round(result.get(\"return_mean\", 0), 4)))
    print()
    print(\"[对战指标摘要] (selfplay_battle_log 最后一条)\")
    print(\"=\" * 70)
    print(\"  奖励:         \" + str(round(result.get(\"reward\", 0), 2)))
    print(\"  回合数:       \" + str(result.get(\"rounds\", 0)))
    print(\"  结果:         \" + str(result.get(\"result\", \"N/A\")))
    print(\"  胜率:         \" + \"{:.2%}\".format(result.get(\"win_rate\", 0)))
    print()
    print(\"=\" * 70)
except Exception as e:
    print(\"解析错误: \" + str(e))
' 2>&1"
}

# 获取时间统计
get_time_statistics() {
    local server_name="$1"
    local model_name="$2"
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local time_file="$latest_dir/system/time_statistics.json"

    local file_check_command="if [ -f '$time_file' ]; then echo 'exists'; else echo 'not_exists'; fi"
    local file_exists=$(execute_remote_command "$server_name" "$file_check_command")

    if [[ "$file_exists" == "exists" ]]; then
        echo "时间统计文件: $time_file"
        echo ""
        local command="cat << 'PYEOF' | /usr/bin/env /root/miniconda3/bin/python
import json
data = json.load(open('$time_file'))
print('=' * 80)
print('训练开始时间: {}'.format(data.get('training_start', '')))
print('训练总时长: {}'.format(data.get('training_time_total', '0') + ' 秒'))
print('平均每轮耗时: {:.2f} 秒'.format(data.get('avg_episode_time', 0)))
print('=' * 80)
PYEOF"
        execute_remote_command "$server_name" "$command"
    else
        echo "状态: ⚠️ 时间统计文件不存在"
    fi
}

# 检查系统资源采样数据
check_system_metrics() {
    local server_name="$1"
    local model_name="$2"
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local metrics_files=(
        "$latest_dir/system/system_metrics.json"
    )

    local metrics_file=""
    for f in "${metrics_files[@]}"; do
        local exists=$(execute_remote_command "$server_name" "if [ -f '$f' ]; then echo 'yes'; else echo 'no'; fi")
        if [[ "$exists" == "yes" ]]; then
            metrics_file="$f"
            break
        fi
    done

    if [[ -z "$metrics_file" ]]; then
        echo_warning "系统资源采样文件不存在，跳过"
        return 0
    fi

    echo "System resources: $metrics_file"
    echo ""

    execute_remote_command "$server_name" "/root/miniconda3/bin/python -c '
import json

try:
    with open(\"$metrics_file\", \"r\") as f:
        data = json.load(f)

    latest = data.get(\"latest\", {})
    stats = data.get(\"stats\", {})

    print(\"=\" * 60)
    print(\"System Resources Stats\")
    print(\"=\" * 60)
    print(\"CPU Avg: \" + str(round(stats.get(\"cpu_avg\", 0), 1)) + \"%\")
    print(\"CPU Max: \" + str(round(stats.get(\"cpu_max\", 0), 1)) + \"%\")
    print(\"RAM Avg: \" + str(round(stats.get(\"ram_avg\", 0), 1)) + \"%\")
    print(\"RAM Max: \" + str(round(stats.get(\"ram_max\", 0), 1)) + \"%\")
    if \"gpu_avg\" in stats:
        print(\"GPU Avg: \" + str(round(stats.get(\"gpu_avg\", 0), 1)) + \"%\")
        print(\"GPU Max: \" + str(round(stats.get(\"gpu_max\", 0), 1)) + \"%\")
    print(\"Samples: \" + str(data.get(\"sample_count\", 0)))

    if latest:
        print()
        print(\"Latest Sample\")
        print(\"=\" * 60)
        print(\"Timestamp: \" + str(latest.get(\"timestamp\", \"\")))
        print(\"Phase: \" + str(latest.get(\"phase\", \"\")))
        print(\"CPU: \" + str(round(latest.get(\"cpu_percent\", 0), 1)) + \"%\")
        ram_used = round(latest.get(\"ram_used_gb\", 0), 1)
        ram_total = round(latest.get(\"ram_total_gb\", 0), 1)
        print(\"RAM: \" + str(round(latest.get(\"ram_percent\", 0), 1)) + \"% (\" + str(ram_used) + \"/\" + str(ram_total) + \"GB)\")
        if \"gpu_util\" in latest:
            print(\"GPU Util: \" + str(latest.get(\"gpu_util\", 0)) + \"%\")
            print(\"GPU Mem: \" + str(latest.get(\"gpu_mem_used_mb\", 0)) + \"/\" + str(latest.get(\"gpu_mem_total_mb\", 0)) + \"MB\")
            print(\"GPU Temp: \" + str(latest.get(\"gpu_temp\", 0)) + \"C\")

    print(\"=\" * 60)

except Exception as e:
    print(\"Error: \" + str(e))
' 2>&1"
}

check_error_log() {
    local server_name="$1"
    local model_name="$2"
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local error_log_file="$latest_dir/training/error.log"

    local file_check_command="if [ -f '$error_log_file' ]; then echo 'exists'; else echo 'not_exists'; fi"
    local file_exists=$(execute_remote_command "$server_name" "$file_check_command")

    if [[ "$file_exists" == "exists" ]]; then
        local line_count=$(execute_remote_command "$server_name" "wc -l < '$error_log_file'")
        echo "异常记录文件: $error_log_file"
        echo "异常记录总数: $line_count"
        echo ""
        echo "最近10条异常记录:";
        local command="cat << 'PYEOF' | /usr/bin/env /root/miniconda3/bin/python
import json
try:
    lines = open('$error_log_file').readlines()
    entries = [json.loads(l) for l in lines[-10:]]
    print('=' * 100)
    for e in entries:
        print('{:20} | {:15} | {}'.format(e.get('timestamp', ''), e.get('exception_type', ''), str(e.get('message', ''))[:50]))
    print('=' * 100)
except Exception as e:
    print('解析异常日志失败:', str(e))
PYEOF"
        execute_remote_command "$server_name" "$command"
    else
        echo "状态: ✅ 无异常记录 (error.log 不存在)"
    fi
}

# 获取训练状态（从 training_status.json 读取关键指标）
get_training_status_summary() {
    local server_name="$1"
    local model_name="$2"
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local status_file="$latest_dir/training/training_status.json"

    local file_check_command="if [ -f '$status_file' ]; then echo 'exists'; else echo 'not_exists'; fi"
    local file_exists=$(execute_remote_command "$server_name" "$file_check_command")

    if [[ "$file_exists" == "exists" ]]; then
        echo "训练状态文件: $status_file"
        echo ""
        local command="cat << 'PYEOF' | /usr/bin/env /root/miniconda3/bin/python
import json
data = json.load(open('$status_file'))
print('=' * 80)
print('当前轮次: {:5} | 总步数: {:10}'.format(data.get('episode', 0), data.get('total_steps', 0)))
print('最佳平均奖励: {:10.2f}'.format(data.get('best_avg_reward', 0)))
print('最近100轮平均奖励: {:10.2f} | 最近10轮平均奖励: {:10.2f}'.format(data.get('avg_reward_last_100', 0), data.get('avg_reward_last_10', 0)))
print('最近100轮平均损失: {:14.6f}'.format(data.get('avg_loss_last_100', 0)))
print('平均回合数: {:10.2f}'.format(data.get('avg_rounds', 0)))
print('训练速度: {:10.2f} 轮/分钟'.format(data.get('training_speed', 0)))
print('-' * 80)
print('对战统计 - 胜: {:4} | 负: {:4} | 平: {:4} | 胜率: {:6.1f}%'.format(
    data.get('total_wins', 0), data.get('total_losses', 0),
    data.get('total_draws', 0), data.get('win_rate', 0) * 100))
print('=' * 80)
ppo = data.get('ppo_metrics', {})
if ppo:
    print('PPO指标 - policy_loss: {:12.6f} | value_loss: {:12.6f} | entropy: {:10.4f}'.format(
        ppo.get('policy_loss', 0), ppo.get('value_loss', 0), ppo.get('entropy', 0)))
stability = data.get('training_stability', {})
if stability:
    print('稳定性 - reward_std: {:10.4f} | loss_std: {:10.6f}'.format(
        stability.get('reward_std', 0), stability.get('loss_std', 0)))
print('=' * 80)
PYEOF"
        execute_remote_command "$server_name" "$command"
    else
        echo "状态: ⚠️ 训练状态文件不存在"
    fi
}

# 获取对战统计（从 episode_batch_baseline_winrates.jsonl 读取 — ppo_v2 适配）
get_battle_stats_summary() {
    local server_name="$1"
    local model_name="$2"
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local winrates_file="${latest_dir}evaluation/episode_batch_baseline_winrates.jsonl"

    local file_check_command="if [ -f '$winrates_file' ]; then echo 'exists'; else echo 'not_exists'; fi"
    local file_exists=$(execute_remote_command "$server_name" "$file_check_command")

    if [[ "$file_exists" == "exists" ]]; then
        echo "对战统计文件: $winrates_file"
        echo ""
        local command="/root/miniconda3/bin/python -c '
import json
with open(\"$winrates_file\", \"r\") as f:
    for line in f:
        pass
    last = json.loads(line)
baselines = last.get(\"baseline_winrates\", {})
print(\"=\" * 80)
print(\"Baseline 对战统计 (最后聚合窗口)\")
print(\"-\" * 80)
print(\"{:20} {:10} {:10} {:10} {:10} {:10}\".format(\"Agent\", \"胜\", \"负\", \"平\", \"场次\", \"胜率\"))
print(\"-\" * 80)
for name, stats in baselines.items():
    print(\"{:20} {:10} {:10} {:10} {:10} {:10.1f}%\".format(
        name[:20], stats.get(\"wins\", 0), stats.get(\"losses\", 0),
        stats.get(\"draws\", 0), stats.get(\"battles\", 0),
        stats.get(\"win_rate\", 0) * 100))
print(\"=\" * 80)
'"
        execute_remote_command "$server_name" "$command"
    else
        echo "状态: ⚠️ 对战统计文件不存在"
    fi
}

# 综合训练进度检查（优化版 - 单一 SSH 连接）
train_progress_check() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")

    echo "=========================================="
    echo "   训练进度检查 (优化版)"
    echo "=========================================="
    echo ""
    echo "服务器: $server_name"
    echo "模型: $model_name"
    echo "模型目录: $model_dir"
    echo ""

    echo "=========================================="
    echo "   正在采集数据..."
    echo "=========================================="

    # 获取远程聚合脚本路径
    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.remote_collect.sh"

    # 上传远程聚合脚本（如果尚未上传）
    local script_exists
    script_exists=$(execute_remote_command "$server_name" "if [ -f '$remote_script_path' ]; then echo 'yes'; else echo 'no'; fi")

    if [[ "$script_exists" != "yes" ]]; then
        upload_file_ssh "$server_name" "$script_dir/remote_collect.sh" "$remote_script_path"
    fi

    # 通过单一 SSH 连接执行远程聚合脚本
    local json_output
    local exit_code

    json_output=$(execute_remote_command "$server_name" "bash '$remote_script_path' '$server_name' '$model_name'")
    exit_code=$?

    if [[ $exit_code -eq 2 ]]; then
        echo "状态: ❌ $json_output"
        return 1
    fi

    # 使用临时文件传递 JSON 数据给 Python 解析器
    local temp_json_file
    temp_json_file=$(mktemp)
    echo "$json_output" > "$temp_json_file"

    # 调用 Python 脚本格式化输出
    python3 "${script_dir}/format_progress.py" "$temp_json_file"
    local python_exit=$?

    # 清理临时文件
    rm -f "$temp_json_file"

    return $python_exit
}

# 从 training_start.log 读取 n_envs 参数
# 每个 EpisodeBatch 记录天然覆盖 n_envs 局，因此 interval=1 就对应"每个 n_envs 周期"
get_n_envs_from_training_log() {
    local server_name="$1"
    local latest_dir="$2"

    local n_envs=$(execute_remote_command "$server_name" \
        "grep '^n_envs:' '${latest_dir}training/training_start.log' 2>/dev/null | head -1 | awk '{print \$2}'")
    if [[ -z "$n_envs" || "$n_envs" == "N/A" ]]; then
        echo "20"  # 默认值
        return 0
    fi
    # 验证是否为数字
    if [[ "$n_envs" =~ ^[0-9]+$ ]]; then
        echo "$n_envs"
    else
        echo "20"
    fi
}

# 获取训练指标完整参数表（SelfPlay 技术参数变化 — ppo_v2 适配）
get_training_metrics_table() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local n_envs=$(get_n_envs_from_training_log "$server_name" "$latest_dir")
    # 每个 EpisodeBatch = n_envs 局，interval=1 即每个 n_envs 周期一行
    local display_interval=1

    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.format_metrics_table.py"

    # 上传脚本
    upload_file_ssh "$server_name" "$script_dir/format_metrics_table.py" "$remote_script_path"

    # 执行远程脚本
    echo "SelfPlay 技术参数表 (n_envs=${n_envs}):"
    echo ""
    execute_remote_command "$server_name" "/root/miniconda3/bin/python '$remote_script_path' \
        '--interval' '${display_interval}' \
        '--train-stats' '${latest_dir}training/episode_batch_train_stats.jsonl' \
        '--battle-stats' '${latest_dir}training/episode_batch_battle_stats.jsonl'"
}

# 获取每局各动作执行次数统计表（ppo_v2 适配：episode_batch_battle_stats.jsonl）
get_action_table() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local n_envs=$(get_n_envs_from_training_log "$server_name" "$latest_dir")
    local display_interval=1

    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.format_action_table.py"

    # 上传脚本
    upload_file_ssh "$server_name" "$script_dir/format_action_table.py" "$remote_script_path"

    # 执行远程脚本
    echo "动作统计表 (n_envs=${n_envs}):"
    echo ""
    execute_remote_command "$server_name" "/root/miniconda3/bin/python '$remote_script_path' \
        '--interval' '${display_interval}' \
        '--battle-stats' '${latest_dir}training/episode_batch_battle_stats.jsonl'"
}

# 获取 Action 类型奖励统计表（ppo_v2 适配：episode_batch_battle_stats.jsonl）
get_action_reward_table() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local n_envs=$(get_n_envs_from_training_log "$server_name" "$latest_dir")
    local display_interval=1

    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.format_action_reward_table.py"

    # 上传脚本
    upload_file_ssh "$server_name" "$script_dir/format_action_reward_table.py" "$remote_script_path"

    # 执行远程脚本
    echo "Action Reward 统计表 (n_envs=${n_envs}):"
    echo ""
    execute_remote_command "$server_name" "/root/miniconda3/bin/python '$remote_script_path' \
        '--interval' '${display_interval}' \
        '--battle-stats' '${latest_dir}training/episode_batch_battle_stats.jsonl'"
}

# 获取 Action Reward 原始数值表（原始数值版，非百分比）
get_action_reward_val_table() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local n_envs=$(get_n_envs_from_training_log "$server_name" "$latest_dir")
    local display_interval=1

    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.format_action_reward_val_table.py"

    # 上传脚本
    upload_file_ssh "$server_name" "$script_dir/format_action_reward_val_table.py" "$remote_script_path"

    # 执行远程脚本
    echo "Action Reward 原始数值表 (n_envs=${n_envs}):"
    echo ""
    execute_remote_command "$server_name" "/root/miniconda3/bin/python '$remote_script_path' \
        '--interval' '${display_interval}' \
        '--battle-stats' '${latest_dir}training/episode_batch_battle_stats.jsonl'"
}

# 获取 Baseline 对战横表（ppo_v2 适配：eval_battle_log.jsonl + episode_batch_baseline_winrates.jsonl）
get_battle_table() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local n_envs=$(get_n_envs_from_training_log "$server_name" "$latest_dir")
    local display_interval=1

    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.format_battle_table.py"

    # 上传脚本
    upload_file_ssh "$server_name" "$script_dir/format_battle_table.py" "$remote_script_path"

    # 执行远程脚本
    echo "Baseline 对战横表 (n_envs=${n_envs}):"
    echo ""
    execute_remote_command "$server_name" "/root/miniconda3/bin/python '$remote_script_path' \
        '--eval-log' '${latest_dir}evaluation/eval_battle_log.jsonl' \
        '--winrates' '${latest_dir}evaluation/episode_batch_baseline_winrates.jsonl' \
        '--interval' '${display_interval}' \
        '--n-envs' '${n_envs}'"
}

# 获取对手池统计表
get_opponent_table() {
    local server_name="$1"
    local model_name="$2"
    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        return 1
    fi

    local league_dir="${latest_dir}checkpoint/league_state/"
    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.format_opponent_table.py"

    local dir_exists
    dir_exists=$(execute_remote_command "$server_name" "if [ -d '$league_dir' ]; then echo 'yes'; else echo 'no'; fi")
    if [[ "$dir_exists" != "yes" ]]; then
        echo_error "league_state 目录不存在 (训练尚未进行到首次保存)"
        return 1
    fi

    upload_file_ssh "$server_name" "$script_dir/format_opponent_table.py" "$remote_script_path"

    execute_remote_command "$server_name" "/root/miniconda3/bin/python '$remote_script_path' '$latest_dir'"
}

# ============================================
# 训练状态检查（轻量级）
# ============================================
exec_train_status() {
    local server="$1"
    local model="$2"
    shift 2

    local VERBOSE=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                break
                ;;
        esac
    done

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    echo_info "检查训练状态..."
    local status=$(get_training_status "$server" "$model")
    echo "训练状态: $status"

    if [[ $VERBOSE == true ]]; then
        echo ""
        echo_info "=== 详细信息 ==="

        local latest_dir=$(get_latest_training_dir "$server" "$model")
        if [[ -n "$latest_dir" ]]; then
            echo "最新训练目录: $latest_dir"

            echo ""
            echo_info "最近的训练日志:"
            get_training_log "$server" "$model"

            echo ""
            echo_info "训练统计信息:"
            get_training_stats "$server" "$model"
        else
            echo_warning "未找到训练目录"
        fi
    fi
}