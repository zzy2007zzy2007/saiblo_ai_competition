#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> train_resume
# ================================================================================

# 恢复训练脚本

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/python_env.sh"

# 参数
SERVER="$1"
MODEL="$2"
shift 2

# 训练参数
EPISODES="10000"
BATCH_SIZE="256"
NUM_MINI_BATCH="4"
UPDATE_EPOCHS="4"
N_ENVS="4"
LEARNING_RATE="0.0003"
GAMMA="0.99"
LAMBDA="0.95"
MODEL_SAVE_INTERVAL="100"
CHECKPOINT_INTERVAL="200"
CHECKPOINT_PATH=""

# 显示用法
show_usage() {
    echo "用法: $0 <server> <model> [参数]"
    echo ""
    echo "参数:"
    echo "  --episodes <num>             训练总轮次"
    echo "  --batch-size <num>           批量大小"
    echo "  --n-envs <num>               训练时的并行环境数量"
    echo "  --learning-rate <num>        学习率"
    echo "  --gamma <num>                折扣因子"
    echo "  --lambda <num>               GAE 参数"
    echo "  --model-save-interval <num>  模型保存间隔（轮次）"
    echo "  --checkpoint-interval <num>  检查点保存间隔（轮次）"
    echo "  --checkpoint <path>          检查点路径"
    echo "  --help                       显示此帮助信息"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --n-envs)
            N_ENVS="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gamma)
            GAMMA="$2"
            shift 2
            ;;
        --lambda)
            LAMBDA="$2"
            shift 2
            ;;
        --model-save-interval)
            MODEL_SAVE_INTERVAL="$2"
            shift 2
            ;;
        --checkpoint-interval)
            CHECKPOINT_INTERVAL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo_error "未知选项: $1"
            show_usage
            exit 1
            ;;
    esac
done

# 检查服务器连接
check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

# 检查是否已有训练在运行
if check_training_running "$SERVER" "$MODEL"; then
    echo_error "训练已在运行中，请先停止现有训练"
    exit 1
fi

# 获取模型目录
MODEL_DIR=$(get_model_dir "$SERVER" "$MODEL")

# 如果没有指定检查点，查找最新的
if [[ -z "$CHECKPOINT_PATH" ]]; then
    CHECKPOINT_PATH=$(execute_remote_command "$SERVER" "ls -t $MODEL_DIR/checkpoints/*.pt 2>/dev/null | head -1")
    if [[ -z "$CHECKPOINT_PATH" ]]; then
        echo_error "未找到可用的检查点"
        exit 1
    fi
    echo_info "使用检查点: $CHECKPOINT_PATH"
fi

# Screen 会话名称
SCREEN_SESSION="${MODEL}_training"

# 构建训练命令
TRAIN_CMD="cd $MODEL_DIR && screen -dmS $SCREEN_SESSION $PYTHON_CMD train.py \
    --episodes $EPISODES \
    --batch-size $BATCH_SIZE \
    --n-envs $N_ENVS \
    --learning-rate $LEARNING_RATE \
    --gamma $GAMMA \
    --lambda $LAMBDA \
    --model-save-interval $MODEL_SAVE_INTERVAL \
    --checkpoint-interval $CHECKPOINT_INTERVAL \
    --checkpoint $CHECKPOINT_PATH"

# 启动训练
echo_info "恢复训练..."
echo_info "服务器: $SERVER"
echo_info "模型: $MODEL"
echo_info "检查点: $CHECKPOINT_PATH"
echo_info "会话: $SCREEN_SESSION"

execute_remote_command "$SERVER" "$TRAIN_CMD"

if [[ $? -eq 0 ]]; then
    echo_success "训练已恢复！"
    echo_info "使用 'bash tools/manager.sh --server $SERVER --model $MODEL train_status' 查看状态"
else
    echo_error "训练恢复失败"
    exit 1
fi