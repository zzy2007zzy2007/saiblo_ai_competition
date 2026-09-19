#!/bin/bash

# ================================================================================
# ⚠️  重要提示：请勿直接调用此脚本！
# ================================================================================
# 此脚本是 tools/manager.sh 的内部实现脚本，不应直接调用。
#
# 正确的使用方式：
#   bash tools/manager.sh --server <server_name> --model <model_name> scan_anomalies
# ================================================================================

# 训练异常扫描脚本
# 扫描 batch_metrics.jsonl / selfplay_battle_log.jsonl / training_*.log
# 发现 NaN、ratio 异常、熵坍塌、梯度爆炸等问题

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/status_manager.sh"

exec_scan_anomalies() {
    local server_name="$1"
    local model_name="$2"

    # 检查服务器连接
    check_server_connection "$server_name"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    local model_dir=$(get_model_dir "$server_name" "$model_name")
    local latest_dir=$(get_latest_training_dir "$server_name" "$model_name")

    if [[ -z "$latest_dir" ]]; then
        echo_error "无法找到训练目录"
        exit 1
    fi

    local training_dir="${latest_dir}training"

    echo "=========================================="
    echo "   训练异常扫描"
    echo "=========================================="
    echo ""
    echo_info "服务器: $server_name"
    echo_info "模型: $model_name"
    echo_info "训练目录: $training_dir"
    echo ""

    # 读取本地 YAML 配置中的阈值参数
    local config_path="${TOOLS_DIR}/../ppo_v2/src/ppo_antwar/configs/ppo_antwar.yaml"
    local max_grad_norm="2.5"
    local max_grad_norm_vf="1000.0"
    if [[ -f "$config_path" ]]; then
        max_grad_norm=$(python3 -c "
import yaml
with open('$config_path') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('ppo', {}).get('max_grad_norm', 2.5))
" 2>/dev/null || echo "2.5")
        max_grad_norm_vf=$(python3 -c "
import yaml
with open('$config_path') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('ppo', {}).get('max_grad_norm_vf', 1000.0))
" 2>/dev/null || echo "1000.0")
    fi

    # 上传扫描脚本到远程服务器（使用 tools/monitor/ 路径，不 hardcode 模型名）
    echo_info "上传扫描脚本..."
    local script_dir="${TOOLS_DIR}/monitor"
    local remote_script_path="${model_dir}/.anomaly_scanner.py"
    local local_script="${script_dir}/_anomaly_scanner.py"

    if [[ ! -f "$local_script" ]]; then
        echo_error "扫描脚本不存在: $local_script"
        exit 1
    fi
    upload_file_ssh "$server_name" "$local_script" "$remote_script_path"
    if [[ $? -ne 0 ]]; then
        echo_error "脚本上传失败"
        exit 1
    fi
    echo_success "脚本上传完成"

    echo ""
    echo_info "正在扫描异常..."

    # 在远程执行扫描（传递配置阈值）
    local json_output
    local exit_code
    json_output=$(execute_remote_command "$server_name" \
        "/root/miniconda3/bin/python '$remote_script_path' '$training_dir' \
         --max_grad_norm '$max_grad_norm' --max_grad_norm_vf '$max_grad_norm_vf' 2>&1")
    exit_code=$?

    echo ""

    if [[ $exit_code -ne 0 ]]; then
        echo_error "扫描执行失败"
        echo "$json_output"
        exit 1
    fi

    # 解析和格式化输出
    local critical_count=$(echo "$json_output" | python3 -c "import sys,json; items=json.load(sys.stdin); print(sum(1 for i in items if i.get('severity')=='CRITICAL'))" 2>/dev/null)
    local warn_count=$(echo "$json_output" | python3 -c "import sys,json; items=json.load(sys.stdin); print(sum(1 for i in items if i.get('severity')=='WARN'))" 2>/dev/null)
    local info_count=$(echo "$json_output" | python3 -c "import sys,json; items=json.load(sys.stdin); print(sum(1 for i in items if i.get('severity')=='INFO'))" 2>/dev/null)

    # 格式化输出
    echo "=========================================="
    echo -e "   扫描结果: \033[31mCRITICAL=${critical_count:-0}\033[0m \033[33mWARN=${warn_count:-0}\033[0m \033[36mINFO=${info_count:-0}\033[0m"
    echo "=========================================="
    echo ""

    # 逐条输出
    echo "$json_output" | python3 -c "
import sys, json
try:
    items = json.load(sys.stdin)
except:
    print('解析输出失败')
    print(sys.stdin.read())
    sys.exit(1)

for item in items:
    sev = item.get('severity', 'INFO')
    typ = item.get('type', 'unknown')
    msg = item.get('msg', '')

    if sev == 'CRITICAL':
        prefix = '🔴 [CRITICAL]'
    elif sev == 'WARN':
        prefix = '🟡 [WARN]    '
    elif sev == 'INFO' and typ in ('summary', 'scan_summary', 'total_time', 'clean'):
        prefix = 'ℹ️  [INFO]    '
    else:
        continue  # skip other INFO items

    print(f'{prefix} {typ}: {msg}')
" 2>&1

    echo ""

    # 清理远程脚本
    execute_remote_command "$server_name" "rm -f '$remote_script_path'" >/dev/null 2>&1
}
