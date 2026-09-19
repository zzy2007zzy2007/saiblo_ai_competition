#!/bin/bash

# 重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> train train_start
# ================================================================================

# 启动训练脚本

exec_train_start() {
    local SERVER="$1"
    local MODEL="$2"
    shift 2

    # 加载配置
    source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
    source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"
    source "$(dirname "${BASH_SOURCE[0]}")/../lib/python_env.sh"

    # 训练参数（默认值）
    local EPISODES="10000"
    local BATCH_SIZE="256"
    local NUM_MINI_BATCH="4"
    local UPDATE_EPOCHS="4"
    local N_ENVS="20"
    local LEARNING_RATE=""
    local GAMMA=""
    local LAMBDA=""
    local MODEL_SAVE_INTERVAL="100"
    local CONFIG_FILE=""
    
    # SelfPlay 特定参数
    local OPPONENT_UPDATE_INTERVAL=""
    local BATTLE_INTERVAL=""
    local N_BATTLES=""
    local OPPONENT_POOL_SIZE=""
    local MIN_OPPONENT_GAMES=""
    local EXPLOIT_PROB=""
    local EXPLORE_PROB=""
    local BASELINE_AGENTS=""
    local WARMUP_EPISODES=""
    local TRANSITION_EPISODES=""

    # 显示用法
    show_usage() {
        echo "用法：$0 <server> <model> [参数]"
        echo ""
        echo "重要参数（命令行）:"
        echo "  --episodes <num>          训练总轮次"
        echo "  --batch-size <num>        批量大小"
        echo "  --n-envs <num>            训练时的并行环境数量"
        echo "  --learning-rate <num>     学习率"
        echo "  --gamma <num>             折扣因子"
        echo "  --gae-lambda <num>        GAE 参数（同 --lambda）"
        echo "  --save-interval <num>       模型保存间隔（同 --model-save-interval）"
        echo ""
        echo "SelfPlay 特定参数:"
        echo "  --opponent-update-interval <num>  对手更新间隔"
        echo "  --battle-interval <num>           对战评估间隔"
        echo "  --n-battles <num>                 每次评估的对战次数"
        echo "  --opponent-pool-size <num>        对手池大小"
        echo "  --min-opponent-games <num>        每个对手的最小对局数"
        echo "  --exploit-prob <num>              对手选择的探索概率"
        echo "  --explore-prob <num>              对手选择的探索概率"
        echo "  --baseline-agents <agent1 agent2>  Baseline agents列表"
        echo ""
        echo "配置文件参数（详细参数）:"
        echo "  --config <file>           配置文件路径（YAML 格式）"
        echo ""
        echo "其他参数:"
        echo "  --help                   显示此帮助信息"
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
            --lambda|--gae-lambda)
                LAMBDA="$2"
                shift 2
                ;;
            --model-save-interval|--save-interval)
                MODEL_SAVE_INTERVAL="$2"
                shift 2
                ;;
            --opponent-update-interval)
                OPPONENT_UPDATE_INTERVAL="$2"
                shift 2
                ;;
            --battle-interval)
                BATTLE_INTERVAL="$2"
                shift 2
                ;;
            --n-battles)
                N_BATTLES="$2"
                shift 2
                ;;
            --opponent-pool-size)
                OPPONENT_POOL_SIZE="$2"
                shift 2
                ;;
            --min-opponent-games)
                MIN_OPPONENT_GAMES="$2"
                shift 2
                ;;
            --exploit-prob)
                EXPLOIT_PROB="$2"
                shift 2
                ;;
            --explore-prob)
                EXPLORE_PROB="$2"
                shift 2
                ;;
            --baseline-agents)
                # 收集所有的 agent 直到遇到另一个 -- 选项
                shift
                BASELINE_AGENTS=""
                while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                    BASELINE_AGENTS="$BASELINE_AGENTS $1"
                    shift
                done
                BASELINE_AGENTS=$(echo "$BASELINE_AGENTS" | xargs)  # 去除首尾空格
                ;;
            --warmup-episodes)
                WARMUP_EPISODES="$2"
                shift 2
                ;;
            --transition-episodes)
                TRANSITION_EPISODES="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
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
    local MODEL_DIR=$(get_model_dir "$SERVER" "$MODEL")

    # 检查训练目录是否存在
    if ! check_training_dir "$SERVER" "$MODEL"; then
        echo_info "模型目录不存在，创建目录: $MODEL_DIR"
        execute_remote_command "$SERVER" "mkdir -p $MODEL_DIR"
    fi

    # Screen 会话名称
    local SCREEN_SESSION="${MODEL}_training"

    # 构建训练命令
    local TRAIN_CMD="screen -dmS $SCREEN_SESSION bash -c 'export PATH=/root/miniconda3/bin:\$PATH && export PYTHONPATH=${MODEL_DIR}/src:/root/autodl-tmp/AntWar/Ant-Game && cd $MODEL_DIR && python3 train.py \
        --episodes $EPISODES \
        --batch-size $BATCH_SIZE \
        --n-envs $N_ENVS \
        --save-interval $MODEL_SAVE_INTERVAL"

    if [[ -n "$LEARNING_RATE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --learning-rate $LEARNING_RATE"
    fi

    # 添加 PPO 参数（如果已设置）
    if [[ -n "$GAMMA" ]]; then
        TRAIN_CMD="$TRAIN_CMD --gamma $GAMMA"
    fi
    if [[ -n "$LAMBDA" ]]; then
        TRAIN_CMD="$TRAIN_CMD --gae-lambda $LAMBDA"
    fi

    # 添加 SelfPlay 参数（如果已设置）
    if [[ -n "$OPPONENT_UPDATE_INTERVAL" ]]; then
        TRAIN_CMD="$TRAIN_CMD --opponent-update-interval $OPPONENT_UPDATE_INTERVAL"
    fi
    if [[ -n "$BATTLE_INTERVAL" ]]; then
        TRAIN_CMD="$TRAIN_CMD --battle-interval $BATTLE_INTERVAL"
    fi
    if [[ -n "$N_BATTLES" ]]; then
        TRAIN_CMD="$TRAIN_CMD --n-battles $N_BATTLES"
    fi
    if [[ -n "$OPPONENT_POOL_SIZE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --opponent-pool-size $OPPONENT_POOL_SIZE"
    fi
    if [[ -n "$MIN_OPPONENT_GAMES" ]]; then
        TRAIN_CMD="$TRAIN_CMD --min-opponent-games $MIN_OPPONENT_GAMES"
    fi
    if [[ -n "$EXPLOIT_PROB" ]]; then
        TRAIN_CMD="$TRAIN_CMD --exploit-prob $EXPLOIT_PROB"
    fi
    if [[ -n "$EXPLORE_PROB" ]]; then
        TRAIN_CMD="$TRAIN_CMD --explore-prob $EXPLORE_PROB"
    fi
    if [[ -n "$BASELINE_AGENTS" ]]; then
        TRAIN_CMD="$TRAIN_CMD --baseline-agents $BASELINE_AGENTS"
    fi
    if [[ -n "$WARMUP_EPISODES" ]]; then
        TRAIN_CMD="$TRAIN_CMD --warmup-episodes $WARMUP_EPISODES"
    fi
    if [[ -n "$TRANSITION_EPISODES" ]]; then
        TRAIN_CMD="$TRAIN_CMD --transition-episodes $TRANSITION_EPISODES"
    fi
    if [[ -n "$CONFIG_FILE" ]]; then
        TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
    fi

    # 启动训练
    echo_info "启动训练..."
    echo_info "服务器: $SERVER"
    echo_info "模型: $MODEL"
    echo_info "会话: $SCREEN_SESSION"

    TRAIN_CMD="$TRAIN_CMD'"

    execute_remote_command "$SERVER" "$TRAIN_CMD"

    if [[ $? -eq 0 ]]; then
        echo_success "训练已启动！"
        echo_info "使用 'bash tools/manager.sh --server $SERVER --model $MODEL train train_status' 查看状态"
    else
        echo_error "训练启动失败"
        exit 1
    fi
}
