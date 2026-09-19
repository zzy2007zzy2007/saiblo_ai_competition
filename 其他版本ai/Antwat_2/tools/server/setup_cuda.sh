#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> setup_cuda
# ================================================================================

# ================================================================================
# CUDA 环境设置脚本
# Usage: bash tools/server/setup_cuda.sh <server> [cuda_version]
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

exec_setup_cuda() {
    local server="$1"
    local cuda_version="${2:-11.8}"

    show_usage() {
        echo "用法: $0 <server> [cuda_version]"
        echo ""
        echo "功能: 设置远程服务器的 CUDA 环境变量"
        echo ""
        echo "参数:"
        echo "  server        服务器名称 (如 server1)"
        echo "  cuda_version  CUDA 版本 (默认: 11.8)"
        echo ""
        echo "示例:"
        echo "  $0 server1"
        echo "  $0 server1 11.7"
        exit 1
    }

    if [[ -z "$server" ]]; then
        show_usage
    fi

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    local cuda_path="/usr/local/cuda-$cuda_version"

    echo_info "检查服务器 $server 上的 CUDA $cuda_version ..."

    local cuda_exists=$(execute_remote_command "$server" "[ -d '$cuda_path' ] && echo 'yes' || echo 'no'")

    if [[ "$cuda_exists" != "yes" ]]; then
        echo_warning "CUDA $cuda_version 未安装在 $cuda_path"

        local available_cuda=$(execute_remote_command "$server" "ls -d /usr/local/cuda-* 2>/dev/null | head -n 1 | xargs basename 2>/dev/null" || echo "")
        if [[ -n "$available_cuda" ]]; then
            echo_info "检测到可用的 CUDA: $available_cuda"
        fi

        echo_info "检查当前 CUDA 环境..."
        execute_remote_command "$server" "echo \$CUDA_HOME"
        execute_remote_command "$server" "which nvcc"
    fi

    echo_info "设置 CUDA $cuda_version 环境变量..."

    local setup_script="cat >> ~/.bashrc << 'EOF'
# CUDA $cuda_version 环境变量
export CUDA_HOME=/usr/local/cuda-$cuda_version
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH
EOF"

    execute_remote_command "$server" "$setup_script"

    echo_success "CUDA 环境设置完成"
    echo_info "请重新登录或在服务器上执行: source ~/.bashrc"

    echo ""
    echo_info "验证 CUDA 安装..."
    execute_remote_command "$server" "nvcc --version 2>/dev/null || echo 'nvcc 不可用'"
}

exec_setup_cuda "$@"