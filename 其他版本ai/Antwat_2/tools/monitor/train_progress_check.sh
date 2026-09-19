#!/bin/bash

# ================================================================================
# ⚠️  重要提示：请勿直接调用此脚本！
# ================================================================================
# 此脚本是 tools/manager.sh 的内部实现脚本，不应直接调用。
#
# 正确的使用方式：
#   bash tools/manager.sh --server <server_name> --model <model_name> train_progress_check
#
# 直接调用此脚本可能导致参数错误、配置不一致或其他问题。
# ================================================================================

# 训练进度综合检查脚本
# 调用 status_manager.sh 中的 train_progress_check 函数

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/status_manager.sh"

# 参数
SERVER="$1"
MODEL="$2"
shift 2

# 检查参数
if [[ -z "$SERVER" || -z "$MODEL" ]]; then
    echo "用法: $0 <server> <model>"
    echo "示例: $0 server2 ppo_v1"
    exit 1
fi

# 检查服务器连接
check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

# 执行综合状态检查（使用新的单一 SSH 连接方式）
train_progress_check "$SERVER" "$MODEL"
