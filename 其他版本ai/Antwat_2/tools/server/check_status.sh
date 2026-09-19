#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> server_status
# ================================================================================

# ================================================================================
# 服务器状态检查脚本
# Usage: bash tools/server/check_status.sh <server>
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

exec_server_status() {
    local server="$1"

    show_usage() {
        echo "用法: $0 <server>"
        echo ""
        echo "功能: 检查服务器各项状态"
        echo ""
        echo "检查项:"
        echo "  - SSH 连接状态"
        echo "  - GPU 状态"
        echo "  - 训练进程状态"
        echo "  - 磁盘空间"
        echo "  - Python 环境"
        echo ""
        echo "示例:"
        echo "  $0 server1"
        exit 1
    }

    if [[ -z "$server" ]]; then
        show_usage
    fi

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    echo "=========================================="
    echo "        服务器状态检查 - $server"
    echo "=========================================="
    echo ""

    echo_info "1. SSH 连接状态..."
    if check_server_connection "$server"; then
        echo_success "SSH 连接正常"
    else
        echo_error "SSH 连接失败"
    fi
    echo ""

    echo_info "2. GPU 状态..."
    GPU_INFO=$(execute_remote_command "$server" "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null" || echo "")
    if [[ -n "$GPU_INFO" ]]; then
        echo_success "GPU 可用"
        echo "   $(echo "$GPU_INFO" | head -n 1)"
    else
        echo_warning "GPU 不可用或 nvidia-smi 未安装"
    fi
    echo ""

    echo_info "3. 训练进程状态..."
    TRAIN_COUNT=$(execute_remote_command "$server" "ps aux | grep -i 'train' | grep -v grep | wc -l" || echo "0")
    echo "   运行中的训练进程: $TRAIN_COUNT"
    if [[ "$TRAIN_COUNT" -gt 0 ]]; then
        echo "   最近启动的进程:"
        execute_remote_command "$server" "ps aux | grep -i 'train' | grep -v grep | head -n 3" | sed 's/^/   /'
    fi
    echo ""

    echo_info "4. 磁盘空间..."
    execute_remote_command "$server" "df -h / | tail -n 1 | awk '{print \"   总空间: \" \$2 \", 已使用: \" \$3 \", 使用率: \" \$5}'"
    echo ""

    echo_info "5. Python 环境..."
    PYTHON_VERSION=$(execute_remote_command "$server" "python3 --version 2>/dev/null" || echo "未安装")
    echo "   $PYTHON_VERSION"

    PIP_VERSION=$(execute_remote_command "$server" "pip3 --version 2>/dev/null" || echo "未安装")
    echo "   $PIP_VERSION"
    echo ""

    echo_info "6. CUDA 版本..."
    CUDA_VERSION=$(execute_remote_command "$server" "nvcc --version 2>/dev/null | grep release | awk '{print \$5}' | sed 's/,//'" || echo "未安装")
    echo "   CUDA: $CUDA_VERSION"
    echo ""

    echo_info "7. 关键目录..."
    ANTWAR_DIR=$(execute_remote_command "$server" "[ -d /root/autodl-tmp/AntWar ] && echo '存在' || echo '不存在'" || echo "未知")
    echo "   /root/autodl-tmp/AntWar: $ANTWAR_DIR"
    echo ""

    echo "=========================================="
    echo "        检查完成"
    echo "=========================================="
}

exec_server_status "$@"