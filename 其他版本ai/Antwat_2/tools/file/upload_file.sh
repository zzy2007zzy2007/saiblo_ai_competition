#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> file_upload <local_path> <remote_path>
# ================================================================================

# ================================================================================
# 文件上传脚本
# Usage: bash tools/file/upload_file.sh <server> <local_path> <remote_path>
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

SERVER="$1"
LOCAL_PATH="$2"
REMOTE_PATH="$3"

show_usage() {
    echo "用法: $0 <server> <local_path> <remote_path>"
    echo ""
    echo "功能: 上传本地文件或目录到远程服务器"
    echo ""
    echo "示例:"
    echo "  $0 server1 ./train.py /root/model/train.py"
    echo "  $0 server1 ./config/ /root/model/config/"
    exit 1
}

if [[ -z "$SERVER" || -z "$LOCAL_PATH" || -z "$REMOTE_PATH" ]]; then
    show_usage
fi

check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

if [[ ! -e "$LOCAL_PATH" ]]; then
    echo_error "本地文件/目录不存在: $LOCAL_PATH"
    exit 1
fi

echo_info "上传 $LOCAL_PATH 到 $SERVER:$REMOTE_PATH"

if [[ -f "$LOCAL_PATH" ]]; then
    upload_file_ssh "$SERVER" "$LOCAL_PATH" "$REMOTE_PATH"
    result=$?
elif [[ -d "$LOCAL_PATH" ]]; then
    upload_file_ssh "$SERVER" "$LOCAL_PATH" "$REMOTE_PATH"
    result=$?
else
    echo_error "不支持的文件类型: $LOCAL_PATH"
    exit 1
fi

if [[ $result -eq 0 ]]; then
    echo_success "上传成功"
else
    echo_error "上传失败"
    exit 1
fi