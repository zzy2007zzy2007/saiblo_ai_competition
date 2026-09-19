#!/bin/bash

# PPO 运维管理工具 v2.0
# 统一管理远程训练服务器的操作

# ================================================================================
#
# 【重要】请勿绕过此脚本直接使用 scp/ssh 命令！
#
# 此脚本封装了所有常用的远程操作，以下功能已有成熟脚本：
#
#   ✓ 文件上传:    file_upload <本地路径> <远程路径>
#   ✓ 文件下载:    file_download <远程路径> <本地路径>
#   ✓ 文件同步:    file_sync <本地路径> <远程路径>
#   ✓ 执行命令:    exec "<远程命令>"
#   ✓ 上传代码:    deploy_code
#   ✓ 启动训练:    train_start [参数]
#   ✓ 停止训练:    train_stop
#   ✓ 查看状态:    train_status
#   ✓ 查看进度:    train_progress_check
#   ✓ 监控GPU:     monitor_gpu
#   ✓ 监控系统:    monitor_system
#   ✓ 服务器连接:  server_connect
#   ✓ 服务器状态:  server_status
#
#
# 常见操作示例：
#
# 【警告】请勿使用以下方式上传文件，它们会绕过 manager.sh 的错误处理：
#   ✗ scp -P port file root@server:/path  (应使用 file_upload)
#   ✗ ssh root@server "cat > file"          (应使用 exec 或 file_upload)
#
# ================================================================================

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 加载配置和工具函数
source "${SCRIPT_DIR}/lib/config_helper.sh"
source "${SCRIPT_DIR}/lib/ssh_helper.sh"
source "${SCRIPT_DIR}/lib/status_manager.sh"

# 全局参数
SERVER=""
MODEL=""

# 显示帮助信息
show_usage() {
    echo "PPO 运维管理工具 v2.0"
    echo ""
    echo "用法: $0 --server <server_name> --model <model_name> <command> [参数]"
    echo ""
    echo "全局参数:"
    echo "  --server <server_name>  指定目标服务器 (server1, server2, server3, server4, server5, server6, server7, server8, server9, server10, server11)"
    echo "  --model <model_name>    指定模型名称 (如 ppo_v1)"
    echo "  --help                 显示此帮助信息"
    echo ""
    echo "训练管理命令:"
    echo "  train_start          启动训练"
    echo "  train_stop           停止训练"
    echo "  train_status         检查训练状态 (-v 详细模式)"
    echo "  train_progress_check 综合检查训练进度"
    echo "  train_metrics_table  查看 SelfPlay 技术参数表"
    echo "  train_action_table   查看每局各动作执行次数统计"
    echo "  train_action_reward_table  查看各类 Action 的 Reward 统计 (占比%)"
    echo "  train_action_reward_val_table  查看各类 Action 的 Reward 统计 (原始数值)"
    echo "  train_battle_table   查看 Baseline 对战横表"
    echo "  train_opponent_table 查看对手池统计"
    echo "  train_report         查看完整训练报告 (参数表+对战表)"
    echo "  scan_anomalies       扫描训练异常 (NaN/ratio/熵坍缩/梯度爆炸)"
    echo "  train_resume         恢复训练"
    echo ""
    echo "部署管理命令:"
    echo "  deploy_code          部署代码 (ppo)"
    echo "  deploy_sdk           部署 SDK (Ant-Game, baselines, 压缩传输)"
    echo "  deploy_battle         部署 battle_single"
    echo ""
    echo "资源监控命令:"
    echo "  monitor_gpu          监控 GPU"
    echo "  monitor_system       监控系统 (--watch 持续监控)"
    echo ""
    echo "文件管理命令:"
    echo "  file_upload          上传文件"
    echo "  file_download        下载文件"
    echo "  file_sync            同步文件"
    echo ""
    echo "服务器管理命令:"
    echo "  server_connect       连接服务器"
    echo "  server_status        服务器状态"
    echo "  server_setup_cuda    设置 CUDA"
    echo "  exec                 执行命令"
    echo ""
    echo "示例:"
    echo "  $0 --server server1 --model ppo_v1 train_start --episodes 10000"
    echo "  $0 --server server1 --model ppo_v1 train_status"
    echo "  $0 --server server1 --model ppo_v1 monitor_gpu --watch"
}

# 解析全局参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --server)
            SERVER="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# 验证必需参数
if [[ -z "$SERVER" ]]; then
    echo_error "缺少参数: --server"
    echo ""
    show_usage
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo_error "缺少参数: --model"
    echo ""
    show_usage
    exit 1
fi

# 验证服务器名称
if ! validate_server "$SERVER"; then
    echo_error "无效的服务器名称: $SERVER"
    echo "可用服务器: ${SERVERS[*]}"
    exit 1
fi

# 获取命令
COMMAND="$1"
shift

# 处理命令
case "$COMMAND" in
    "train_start")
        source "${SCRIPT_DIR}/deploy/start_training.sh"
        exec_train_start "$SERVER" "$MODEL" "$@"
        ;;
    "train_stop")
        source "${SCRIPT_DIR}/deploy/stop_training.sh"
        exec_train_stop "$SERVER" "$MODEL" "$@"
        ;;
    "train_status")
        exec_train_status "$SERVER" "$MODEL" "$@"
        ;;
    "train_progress_check")
        train_progress_check "$SERVER" "$MODEL"
        ;;
    "train_metrics_table")
        get_training_metrics_table "$SERVER" "$MODEL"
        ;;
    "train_action_table")
        get_action_table "$SERVER" "$MODEL"
        ;;
    "train_action_reward_table")
        get_action_reward_table "$SERVER" "$MODEL"
        ;;
    "train_action_reward_val_table")
        get_action_reward_val_table "$SERVER" "$MODEL"
        ;;
    "train_battle_table")
        get_battle_table "$SERVER" "$MODEL"
        ;;
    "train_opponent_table")
        get_opponent_table "$SERVER" "$MODEL"
        ;;
    "scan_anomalies")
        source "${SCRIPT_DIR}/monitor/scan_anomalies.sh"
        exec_scan_anomalies "$SERVER" "$MODEL"
        ;;
    "train_report")
        echo "=========================================="
        echo "   完整训练报告"
        echo "=========================================="
        echo ""
        get_training_metrics_table "$SERVER" "$MODEL"
        echo ""
        echo "------------------------------------------"
        echo ""
        get_battle_table "$SERVER" "$MODEL"
        ;;
    "train_resume")
        source "${SCRIPT_DIR}/deploy/resume_training.sh"
        exec_train_resume "$SERVER" "$MODEL" "$@"
        ;;
    "deploy_code")
        source "${SCRIPT_DIR}/deploy/deploy_code.sh"
        exec_deploy_code "$SERVER" "$MODEL" "$@"
        ;;

    "deploy_sdk")
        source "${SCRIPT_DIR}/deploy/deploy_sdk.sh"
        exec_deploy_sdk "$SERVER" "$MODEL" "$@"
        ;;
    "tensorboard")
        source "${SCRIPT_DIR}/deploy/tensorboard.sh"
        exec_tensorboard "$SERVER" "$MODEL" "$@"
        ;;
    "deploy_battle")
        source "${SCRIPT_DIR}/deploy/deploy_battle.sh"
        exec_deploy_battle "$SERVER" "$MODEL" "$@"
        ;;
    "monitor_gpu")
        source "${SCRIPT_DIR}/monitor/check_gpu.sh"
        exec_monitor_gpu "$SERVER" "$MODEL" "$@"
        ;;
    "monitor_system")
        source "${SCRIPT_DIR}/monitor/check_system.sh"
        exec_check_system "$SERVER" "$@"
        ;;
    "file_upload")
        LOCAL_PATH="$1"
        REMOTE_PATH="$2"
        if [[ -z "$LOCAL_PATH" || -z "$REMOTE_PATH" ]]; then
            echo_error "缺少文件路径参数"
            echo "用法: file_upload <本地路径> <远程路径>"
            exit 1
        fi
        upload_file_ssh "$SERVER" "$LOCAL_PATH" "$REMOTE_PATH"
        if [[ $? -eq 0 ]]; then
            echo_success "文件上传成功"
        else
            echo_error "文件上传失败"
            exit 1
        fi
        ;;
    "file_download")
        REMOTE_PATH="$1"
        LOCAL_PATH="$2"
        if [[ -z "$REMOTE_PATH" || -z "$LOCAL_PATH" ]]; then
            echo_error "缺少文件路径参数"
            echo "用法: file_download <远程路径> <本地路径>"
            exit 1
        fi
        download_file_ssh "$SERVER" "$REMOTE_PATH" "$LOCAL_PATH"
        if [[ $? -eq 0 ]]; then
            echo_success "文件下载成功"
        else
            echo_error "文件下载失败"
            exit 1
        fi
        ;;
    "file_sync")
        LOCAL_PATH="$1"
        REMOTE_PATH="$2"
        if [[ -z "$LOCAL_PATH" || -z "$REMOTE_PATH" ]]; then
            echo_error "缺少文件路径参数"
            echo "用法: file_sync <本地路径> <远程路径>"
            exit 1
        fi
        source "${SCRIPT_DIR}/file/sync_files.sh"
        exec_sync_files "$SERVER" "$LOCAL_PATH" "$REMOTE_PATH"
        ;;
    "exec")
        COMMAND_STR="$1"
        if [[ -z "$COMMAND_STR" ]]; then
            echo_error "缺少命令参数"
            echo "用法: exec '<远程命令>'"
            exit 1
        fi
        execute_remote_command "$SERVER" "$COMMAND_STR"
        ;;
    "server_connect")
        source "${SCRIPT_DIR}/server/connect.sh"
        exec_server_connect "$SERVER"
        ;;
    "server_status")
        source "${SCRIPT_DIR}/server/check_status.sh"
        exec_server_status "$SERVER"
        ;;
    "server_setup_cuda")
        CUDA_VERSION="$1"
        source "${SCRIPT_DIR}/server/setup_cuda.sh"
        exec_setup_cuda "$SERVER" "$CUDA_VERSION"
        ;;
    "")
        echo_error "未指定命令"
        show_usage
        exit 1
        ;;
    *)
        echo_error "未知的命令: $COMMAND"
        show_usage
        exit 1
        ;;
esac
