#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> exec "<command>"
# ================================================================================

# ================================================================================
# 执行远程命令脚本
# Usage: bash tools/server/exec_command.sh <server> "<command>"
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

exec_remote_command() {
    local server="$1"
    local command="$2"

    show_usage() {
        echo "用法: $0 <server> \"<command>\""
        echo ""
        echo "功能: 在远程服务器上执行命令"
        echo ""
        echo "示例:"
        echo "  $0 server1 \"ls -la /root/\""
        echo "  $0 server1 \"ps aux | grep python\""
        exit 1
    }

    if [[ -z "$server" || -z "$command" ]]; then
        show_usage
    fi

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    echo_info "在 $server 上执行命令: $command"
    echo ""

    execute_remote_command "$server" "$command"
    result=$?

    if [[ $result -eq 0 ]]; then
        echo ""
        echo_success "命令执行成功 (退出码: $result)"
    else
        echo ""
        echo_error "命令执行失败 (退出码: $result)"
    fi

    exit $result
}

exec_remote_command "$@"