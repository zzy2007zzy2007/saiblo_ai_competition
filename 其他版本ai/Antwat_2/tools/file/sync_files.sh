#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> file_sync <local_path> <remote_path>
# ================================================================================

# ================================================================================
# 文件同步脚本
# Usage: bash tools/file/sync_files.sh <server> <local_path> <remote_path> [--dry-run]
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

SERVER="$1"
LOCAL_PATH="$2"
REMOTE_PATH="$3"
DRY_RUN="$4"

show_usage() {
    echo "用法: $0 <server> <local_path> <remote_path> [--dry-run]"
    echo ""
    echo "功能: 同步本地和远程文件（双向）"
    echo ""
    echo "选项:"
    echo "  --dry-run    预览同步操作，不执行实际同步"
    echo ""
    echo "示例:"
    echo "  $0 server1 ./train.py /root/model/train.py"
    echo "  $0 server1 ./train.py /root/model/train.py --dry-run"
    exit 1
}

if [[ -z "$SERVER" || -z "$LOCAL_PATH" || -z "$REMOTE_PATH" ]]; then
    show_usage
fi

check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

get_file_info() {
    local server="$1"
    local path="$2"

    local command="if [ -e '$path' ]; then
        if [ -f '$path' ]; then
            echo 'file'
        elif [ -d '$path' ]; then
            echo 'dir'
        else
            echo 'other'
        fi
    else
        echo 'not_exists'
    fi"

    execute_remote_command "$server" "$command"
}

sync_files() {
    local direction="$1"
    local src="$2"
    local dst="$3"

    if [[ ! -e "$src" ]]; then
        echo_error "源路径不存在: $src"
        return 1
    fi

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo_info "[预演] $direction: $src -> $dst"
        return 0
    fi

    echo_info "同步 $direction: $src -> $dst"

    if [[ -f "$src" ]]; then
        upload_file_ssh "$SERVER" "$src" "$dst"
    elif [[ -d "$src" ]]; then
        upload_file_ssh "$SERVER" "$src" "$dst"
    fi

    return $?
}

exec_sync_files() {
    local server="$1"
    local local_path="$2"
    local remote_path="$3"

    show_usage() {
        echo "用法: $0 <server> <local_path> <remote_path> [--dry-run]"
        echo ""
        echo "功能: 同步本地和远程文件（双向）"
        echo ""
        echo "选项:"
        echo "  --dry-run    预览同步操作，不执行实际同步"
        echo ""
        echo "示例:"
        echo "  $0 server1 ./train.py /root/model/train.py"
        echo "  $0 server1 ./train.py /root/model/train.py --dry-run"
        exit 1
    }

    if [[ -z "$server" || -z "$local_path" || -z "$remote_path" ]]; then
        show_usage
    fi

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    SERVER="$server"
    LOCAL_PATH="$local_path"
    REMOTE_PATH="$remote_path"

    echo_info "开始文件同步..."
    echo_info "本地: $LOCAL_PATH"
    echo_info "远程: $SERVER:$REMOTE_PATH"
    echo ""

    local_type="local"
    remote_type=$(get_file_info "$SERVER" "$REMOTE_PATH")

    if [[ "$LOCAL_PATH" == *"/"* ]]; then
        local_basename=$(basename "$LOCAL_PATH")
    else
        local_basename="$LOCAL_PATH"
    fi

    remote_file="${REMOTE_PATH%/}/${local_basename}"

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo_info "=== 同步计划 ==="
        sync_files "上传" "$LOCAL_PATH" "$REMOTE_PATH"
    else
        sync_files "上传" "$LOCAL_PATH" "$REMOTE_PATH"
        result=$?

        if [[ $result -eq 0 ]]; then
            echo_success "同步完成"
        else
            echo_error "同步失败"
            exit 1
        fi
    fi
}

exec_sync_files "$@"