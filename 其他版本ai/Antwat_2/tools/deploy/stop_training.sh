#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> train_stop
# ================================================================================

# 停止训练脚本

exec_train_stop() {
    local SERVER="$1"
    local MODEL="$2"
    shift 2

    # 加载配置（ssh_helper.sh 已包含 config_helper.sh）
    source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

    # 检查服务器连接
    check_server_connection "$SERVER"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    # Screen 会话名称
    local SCREEN_SESSION="${MODEL}_training"

    # 尝试关闭 screen 会话（即使不存在也继续清理残留进程）
    check_session_command="screen -ls | grep -q '$SCREEN_SESSION'"
    if execute_remote_command "$SERVER" "$check_session_command"; then
        echo_info "停止训练会话: $SCREEN_SESSION"
        execute_remote_command "$SERVER" "screen -S $SCREEN_SESSION -X quit 2>/dev/null || true"
    else
        echo_warning "训练会话 $SCREEN_SESSION 不存在，仍尝试清理残留进程"
    fi

    # 清理所有残留进程（screen 退出后子进程可能仍在运行）
    execute_remote_command "$SERVER" "pkill -9 -f 'python3 train.py' 2>/dev/null || true; pkill -9 -f 'multiprocessing.spawn' 2>/dev/null || true; pkill -9 -f 'multiprocessing.resource_tracker' 2>/dev/null || true"

    # 验证清理结果
    sleep 2
    local remaining=$(execute_remote_command "$SERVER" "ps aux | grep -E 'python3 train.py|multiprocessing.spawn' | grep -v grep | wc -l")
    if [[ "$remaining" -gt 0 ]]; then
        echo_warning "仍有 $remaining 个残留进程，已尝试清理"
    else
        echo_success "所有进程已清理"
    fi
}
