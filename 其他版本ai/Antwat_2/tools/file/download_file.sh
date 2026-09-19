#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> file_download <remote_path> <local_path>
# ================================================================================

# ================================================================================
# 文件下载脚本
# Usage: bash tools/file/download_file.sh <server> <remote_path> <local_path>
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

SERVER="$1"
REMOTE_PATH="$2"
LOCAL_PATH="$3"

show_usage() {
    echo "用法: $0 <server> <remote_path> <local_path>"
    echo ""
    echo "功能: 从远程服务器下载文件或目录到本地"
    echo ""
    echo "示例:"
    echo "  $0 server1 /root/model/train.py ./train.py"
    echo "  $0 server1 /root/model/config/ ./config/"
    exit 1
}

if [[ -z "$SERVER" || -z "$REMOTE_PATH" || -z "$LOCAL_PATH" ]]; then
    show_usage
fi

check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

echo_info "从 $SERVER:$REMOTE_PATH 下载到 $LOCAL_PATH"

download_file_ssh "$SERVER" "$REMOTE_PATH" "$LOCAL_PATH"
result=$?

if [[ $result -eq 0 ]]; then
    echo_success "下载成功"
else
    echo_error "下载失败"
    exit 1
fi