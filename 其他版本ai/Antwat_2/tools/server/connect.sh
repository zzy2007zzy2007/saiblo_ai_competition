#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> server_connect
# ================================================================================

# ================================================================================
# 服务器连接脚本
# Usage: bash tools/server/connect.sh <server>
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

SERVER="$1"

show_usage() {
    echo "用法: $0 <server>"
    echo ""
    echo "功能: 连接到远程服务器"
    echo ""
    echo "示例:"
    echo "  $0 server1"
    exit 1
}

if [[ -z "$SERVER" ]]; then
    show_usage
fi

check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

set_server "$SERVER"

echo_info "连接到服务器 $SERVER (${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT})..."
echo_info "输入 exit 退出连接"

sshpass -p "${SERVER_PASSWORD}" ssh -p "${SERVER_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${SERVER_USER}@${SERVER_HOST}"

exec_server_connect() {
    local server="$1"

    show_usage() {
        echo "用法: $0 <server>"
        echo ""
        echo "功能: 连接到远程服务器"
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

    SERVER="$server"
    set_server "$SERVER"

    echo_info "连接到服务器 $SERVER (${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT})..."
    echo_info "输入 exit 退出连接"

    sshpass -p "${SERVER_PASSWORD}" ssh -p "${SERVER_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${SERVER_USER}@${SERVER_HOST}"
}

exec_server_connect "$@"