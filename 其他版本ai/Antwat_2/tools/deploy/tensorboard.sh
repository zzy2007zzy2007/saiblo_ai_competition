#!/bin/bash
# TensorBoard 启动脚本 - 在远程服务器上启动 TensorBoard 并提示端口转发
# 用法: manager.sh --server <server> --model <model> tensorboard [port]

exec_tensorboard() {
    local SERVER="$1"
    local MODEL="$2"
    local PORT="${3:-6006}"
    shift 2
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        PORT="$1"
        shift
    fi

    source "${SCRIPT_DIR}/lib/ssh_helper.sh"

    local MODEL_DIR="${MODEL_BASE_DIR}/${MODEL}"
    local OUTPUT_DIR="${MODEL_BASE_DIR}/outputs"

    set_server "$SERVER" || return 1

    echo "正在查找最新的 TensorBoard 日志目录..."
    LATEST_DIR=$(execute_remote_command "$SERVER" "ls -t ${OUTPUT_DIR} 2>/dev/null | head -1")
    if [[ -z "$LATEST_DIR" ]]; then
        echo "错误: 未找到训练输出目录"
        return 1
    fi
    TB_DIR="${OUTPUT_DIR}/${LATEST_DIR}/tensorboard"
    echo "TensorBoard 日志目录: ${TB_DIR}"

    DIR_EXISTS=$(execute_remote_command "$SERVER" "test -d ${TB_DIR} && echo 'yes' || echo 'no'")
    if [[ "$DIR_EXISTS" != "yes" ]]; then
        echo "警告: TensorBoard 目录不存在 (${TB_DIR})"
        echo "可能训练尚未产生数据，或 tensorboard 未启用"
    fi

    OLD_PID=$(execute_remote_command "$SERVER" "ps aux | grep 'tensorboard' | grep -v grep | awk '{print \$2}' | head -1")
    if [[ -n "$OLD_PID" ]]; then
        echo "发现已有 TensorBoard 进程 (PID: $OLD_PID)，正在停止..."
        execute_remote_command "$SERVER" "kill $OLD_PID 2>/dev/null; sleep 1; kill -9 $OLD_PID 2>/dev/null"
    fi

    echo "正在远程服务器启动 TensorBoard (端口: $PORT)..."
    execute_remote_command "$SERVER" "nohup tensorboard --logdir=${TB_DIR} --port=${PORT} --bind_all > /tmp/tensorboard.log 2>&1 &"
    sleep 2

    TB_PID=$(execute_remote_command "$SERVER" "ps aux | grep 'tensorboard' | grep -v grep | awk '{print \$2}' | head -1")
    if [[ -n "$TB_PID" ]]; then
        echo ""
        echo "============================================"
        echo "  TensorBoard 已启动! (PID: $TB_PID)"
        echo "============================================"
        echo ""
        echo "本地查看步骤:"
        echo ""
        echo "  1. 打开新终端，执行端口转发:"
        echo ""
        echo "     ssh -N -L ${PORT}:localhost:${PORT} -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
        echo ""
        echo "  2. 在浏览器中打开:"
        echo ""
        echo "     http://localhost:${PORT}"
        echo ""
        echo "============================================"
    else
        echo "错误: TensorBoard 启动失败，请检查 /tmp/tensorboard.log"
    fi
}
