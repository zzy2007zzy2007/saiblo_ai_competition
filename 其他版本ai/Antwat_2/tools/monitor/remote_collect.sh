#!/bin/bash

# ================================================================================
# 远程训练进度信息采集聚合脚本
# 通过单一 SSH 连接获取所有训练进度相关信息
# 输出 JSON 格式，便于本地脚本解析
# ================================================================================

set -euo pipefail

# 参数解析
SERVER="${1:-}"
MODEL="${2:-}"

if [[ -z "$SERVER" || -z "$MODEL" ]]; then
    echo '{"status":"failed","error":"Missing parameters: server and model required"}' >&2
    exit 2
fi

# 配置
MODEL_BASE_DIR="/root/autodl-tmp/AntWar"
MODEL_DIR="${MODEL_BASE_DIR}/${MODEL}"
SCREEN_SESSION="${MODEL}_training"
PYTHON_CMD="/root/miniconda3/bin/python"
JSON_OUTPUT=""

# 日志函数（仅调试用）
debug_log() {
    : # 静默模式
}

# 获取最新训练目录（固定结构：outputs/{timestamp}/）
get_latest_training_dir() {
    local latest
    latest=$(ls -td "${MODEL_DIR}/outputs"/20[0-9][0-9][0-9][0-9][0-1][0-9][0-3][0-9]_[0-2][0-9][0-5][0-9][0-5][0-9]/ 2>/dev/null | head -1 || true)
    if [[ -n "$latest" ]]; then
        echo "${latest%/}/"
        return 0
    fi
    echo ""
    return 1
}

# 采集 Screen 会话状态
collect_screen_info() {
    local screen_output
    screen_output=$(screen -ls 2>/dev/null || true)

    if echo "$screen_output" | grep -q "$SCREEN_SESSION"; then
        local session_line
        session_line=$(echo "$screen_output" | grep "$SCREEN_SESSION" | head -1)
        local session_id=$(echo "$session_line" | awk '{print $1}' | sed 's/\..*//')
        local created_at=$(echo "$session_line" | awk '{print $NF}' | tr -d '()')
        local status="Detached"
        if echo "$session_line" | grep -q "Attached"; then
            status="Attached"
        fi

        cat <<EOF
"screen": {
    "exists": true,
    "session_name": "${SCREEN_SESSION}",
    "session_id": "${session_id}",
    "created_at": "${created_at}",
    "status": "${status}"
}
EOF
    else
        cat <<EOF
"screen": {
    "exists": false,
    "session_name": "${SCREEN_SESSION}",
    "session_id": null,
    "created_at": null,
    "status": null
}
EOF
    fi
}

# 采集训练进程信息
collect_process_info() {
    local process_info
    process_info=$(ps aux | grep "python.*train.py" | grep -v grep | head -1 || true)

    if [[ -n "$process_info" ]]; then
        local pid=$(echo "$process_info" | awk '{print $2}')
        local cpu_percent=$(echo "$process_info" | awk '{print $3}')
        local memory_percent=$(echo "$process_info" | awk '{print $4}')
        local vsz=$(echo "$process_info" | awk '{print $5}')
        local rss=$(echo "$process_info" | awk '{print $6}')
        local start_time=$(echo "$process_info" | awk '{print $9}')
        local cmdline=$(echo "$process_info" | awk '{$1=$2=$3=$4=$5=$6=$7=$8=$9=""; print $0}' | xargs)

        local memory_mb=$((rss / 1024))

        cat <<EOF
"process": {
    "pid": ${pid},
    "cpu_percent": ${cpu_percent},
    "memory_percent": ${memory_percent},
    "memory_mb": ${memory_mb},
    "start_time": "${start_time}",
    "cmdline": "${cmdline}"
}
EOF
    else
        cat <<EOF
"process": null
EOF
    fi
}

# 采集训练指标（ppo_v2 适配：batch_metrics.jsonl + selfplay_battle_log.jsonl + episode_batch_battle_stats.jsonl）
collect_training_metrics() {
    local latest_dir="$1"

    "$PYTHON_CMD" -c "
import json

result = {}
try:
    # 从 batch_metrics.jsonl 读取最后一条
    with open('${latest_dir}training/batch_metrics.jsonl') as f:
        for line in f:
            pass
        if line:
            batch = json.loads(line)
            result['current_episode'] = batch.get('episode', 0)
            result['policy_loss'] = batch.get('policy_loss', 0)
            result['value_loss'] = batch.get('value_loss', 0)
            result['entropy'] = batch.get('entropy', 0)
            result['learning_rate'] = batch.get('learning_rate', 0)

    # 从 selfplay_battle_log.jsonl 读取最后一条
    with open('${latest_dir}training/selfplay_battle_log.jsonl') as f:
        for line in f:
            pass
        if line:
            battle = json.loads(line)
            result['avg_reward'] = battle.get('reward', 0)
            result['avg_rounds'] = battle.get('rounds', 0)
            result['last_result'] = battle.get('result', '')

    # 从 episode_batch_battle_stats.jsonl 读取最后的胜率
    with open('${latest_dir}training/episode_batch_battle_stats.jsonl') as f:
        for line in f:
            pass
        if line:
            stats = json.loads(line)
            result['win_rate'] = stats.get('win_rate', 0)

    result['status'] = 'ok'
except FileNotFoundError:
    result['status'] = 'partial'
except Exception as e:
    result = {'error': str(e)}

print(json.dumps(result, ensure_ascii=False))
" 2>/dev/null || echo '{"error": "python parse failed"}'
}

# 采集系统资源采样
collect_system_metrics() {
    local latest_dir="$1"
    local metrics_file="${latest_dir}system/system_metrics.json"

    if [[ -f "$metrics_file" ]]; then
        "$PYTHON_CMD" -c "
import json

try:
    with open('$metrics_file', 'r') as f:
        data = json.load(f)

    stats = data.get('stats', {})
    latest = data.get('latest', {})

    result = {
        'cpu_avg': stats.get('cpu_avg', 0),
        'cpu_max': stats.get('cpu_max', 0),
        'ram_avg': stats.get('ram_avg', 0),
        'ram_max': stats.get('ram_max', 0),
        'gpu_avg': stats.get('gpu_avg', 0),
        'gpu_max': stats.get('gpu_max', 0),
        'samples': data.get('sample_count', 0),
        'latest': {
            'timestamp': latest.get('timestamp', ''),
            'phase': latest.get('phase', ''),
            'cpu_percent': latest.get('cpu_percent', 0),
            'ram_percent': latest.get('ram_percent', 0),
            'ram_used_gb': latest.get('ram_used_gb', 0),
            'ram_total_gb': latest.get('ram_total_gb', 0),
            'gpu_util': latest.get('gpu_util', 0),
            'gpu_mem_mb': latest.get('gpu_mem_used_mb', 0),
            'gpu_temp': latest.get('gpu_temp', 0)
        }
    }

    print(json.dumps(result, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'error': str(e)}, ensure_ascii=False))
" 2>/dev/null || echo '{"error": "python parse failed"}'
    else
        echo '{"error": "file_not_found"}'
    fi
}

# 采集 GPU 状态
collect_gpu_info() {
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || true)

    if [[ -n "$gpu_info" ]]; then
        local name=$(echo "$gpu_info" | cut -d',' -f1 | sed 's/^ *//;s/ *$//')
        local util=$(echo "$gpu_info" | cut -d',' -f2 | sed 's/^ *//;s/ *$//')
        local mem_used=$(echo "$gpu_info" | cut -d',' -f3 | sed 's/^ *//;s/ *$//')
        local mem_total=$(echo "$gpu_info" | cut -d',' -f4 | sed 's/^ *//;s/ *$//')
        local temp=$(echo "$gpu_info" | cut -d',' -f5 | sed 's/^ *//;s/ *$//')

        cat <<EOF
"gpu": {
    "name": "${name}",
    "utilization": ${util},
    "memory_used_mb": ${mem_used},
    "memory_total_mb": ${mem_total},
    "temperature": ${temp}
}
EOF
    else
        cat <<EOF
"gpu": null
EOF
    fi
}

# 采集对战统计（ppo_v2 适配：episode_batch_baseline_winrates.jsonl）
collect_battle_stats() {
    local latest_dir="$1"

    "$PYTHON_CMD" -c "
import json

try:
    with open('${latest_dir}evaluation/episode_batch_baseline_winrates.jsonl') as f:
        for line in f:
            pass
        if line:
            data = json.loads(line)
            result = {'exists': True, 'data': data.get('baseline_winrates', {})}
        else:
            result = {'exists': False, 'error': 'empty_file'}
except FileNotFoundError:
    result = {'exists': False, 'error': 'file_not_found'}
except Exception as e:
    result = {'exists': True, 'error': str(e)}

print(json.dumps(result, ensure_ascii=False))
" 2>/dev/null || echo '{"exists": false, "error": "python parse failed"}'
}

# 采集异常日志
collect_error_log() {
    local latest_dir="$1"
    local error_file="${latest_dir}training/error.log"

    if [[ -f "$error_file" ]]; then
        local count
        count=$(wc -l < "$error_file" 2>/dev/null || echo "0")

        "$PYTHON_CMD" -c "
import json

try:
    with open('$error_file', 'r') as f:
        lines = f.readlines()

    entries = []
    for line in lines[-10:]:
        try:
            entries.append(json.loads(line.strip()))
        except:
            pass

    print(json.dumps({'exists': True, 'count': $count, 'recent': entries}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'exists': True, 'count': $count, 'error': str(e)}, ensure_ascii=False))
" 2>/dev/null || echo "{\"exists\": true, \"count\": 0, \"error\": \"python parse failed\"}"
    else
        echo '{"exists": false, "count": 0}'
    fi
}

# 采集时间统计
collect_time_stats() {
    local latest_dir="$1"
    local time_file="${latest_dir}system/time_statistics.json"

    if [[ -f "$time_file" ]]; then
        "$PYTHON_CMD" -c "
import json

try:
    with open('$time_file', 'r') as f:
        data = json.load(f)

    print(json.dumps({'exists': True, 'data': data}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'exists': True, 'error': str(e)}, ensure_ascii=False))
" 2>/dev/null || echo '{"exists": true, "error": "python parse failed"}'
    else
        echo '{"exists": false, "error": "file_not_found"}'
    fi
}

# 主函数
main() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local current_time_unix=$(date '+%s')

    # 检查训练目录是否存在
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo '{"status":"failed","error":"Model directory not found: '${MODEL_DIR}'","timestamp":"'${timestamp}'"}' >&2
        exit 2
    fi

    # 获取最新训练目录
    local latest_dir
    latest_dir=$(get_latest_training_dir)

    if [[ -z "$latest_dir" ]]; then
        echo '{"status":"failed","error":"No training directory found","timestamp":"'${timestamp}'"}' >&2
        exit 2
    fi

    # 确保目录路径格式正确
    latest_dir="${latest_dir%/}/"

    local has_errors=0
    local error_fields=""

    # 收集各项数据
    local screen_json
    screen_json=$(collect_screen_info)

    local process_json
    process_json=$(collect_process_info)

    local training_metrics_json
    training_metrics_json=$(collect_training_metrics "$latest_dir")
    if echo "$training_metrics_json" | grep -q '"error"'; then
        has_errors=1
        error_fields="${error_fields}training_metrics,"
    fi

    local system_metrics_json
    system_metrics_json=$(collect_system_metrics "$latest_dir")
    if echo "$system_metrics_json" | grep -q '"error"'; then
        has_errors=1
        error_fields="${error_fields}system_metrics,"
    fi

    local gpu_json
    gpu_json=$(collect_gpu_info)

    local battle_stats_json
    battle_stats_json=$(collect_battle_stats "$latest_dir")

    local error_log_json
    error_log_json=$(collect_error_log "$latest_dir")

    local time_stats_json
    time_stats_json=$(collect_time_stats "$latest_dir")

    # 组装最终 JSON
    local final_status="success"
    if [[ $has_errors -eq 1 ]]; then
        final_status="partial"
    fi

    cat <<EOF
{
    "status": "${final_status}",
    "timestamp": "${timestamp}",
    "server": "${SERVER}",
    "model": "${MODEL}",
    "training_dir": "${latest_dir}",
    ${screen_json},
    ${process_json},
    "training_metrics": ${training_metrics_json},
    "system_metrics": ${system_metrics_json},
    ${gpu_json},
    "battle_stats": ${battle_stats_json},
    "error_log": ${error_log_json},
    "time_stats": ${time_stats_json}
}
EOF

    if [[ $has_errors -eq 1 ]]; then
        exit 1
    fi
    exit 0
}

main
