#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> monitor_gpu
# ================================================================================

# 检查 GPU 使用情况脚本

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

# 参数
SERVER="$1"
MODEL="$2"
shift 2

# 选项
WATCH=false

# 显示用法
show_usage() {
    echo "用法: $0 <server> <model> [参数]"
    echo ""
    echo "参数:"
    echo "  -w, --watch          持续监控"
    echo "  --help               显示此帮助信息"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -w|--watch)
            WATCH=true
            shift
            ;;
        *)
            echo "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查服务器连接
check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

# 检查 GPU 命令
gpu_command="nvidia-smi"

# 持续监控
if [[ $WATCH == true ]]; then
    echo_info "持续监控 GPU 使用情况..."
    echo_info "按 Ctrl+C 退出"

    while true; do
        echo ""
        echo "========================================"
        echo "时间: $(date)"
        echo "========================================"
        execute_remote_command "$SERVER" "$gpu_command"
        sleep 5
    done
else
    # 单次检查
    echo_info "检查 GPU 使用情况..."
    execute_remote_command "$SERVER" "$gpu_command"
fi