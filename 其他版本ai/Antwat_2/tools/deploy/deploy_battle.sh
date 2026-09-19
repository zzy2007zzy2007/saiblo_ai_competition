#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> deploy_battle
# ================================================================================

# 部署 battle_single 脚本

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

# 获取项目根目录
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "$(cd "${script_dir}/.." && pwd)"
}

exec_deploy_battle() {
    local server="$1"
    local model="$2"

    show_usage() {
        echo "用法: $0 <server> <model>"
        echo ""
        echo "功能: 部署 battle_single 代码到远程服务器"
        echo ""
        echo "部署内容:"
        echo "  - battle_single 目录"
        echo ""
        echo "示例:"
        echo "  $0 server5 ppo_v1"
        exit 1
    }

    if [[ -z "$server" || -z "$model" ]]; then
        show_usage
    fi

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    PROJECT_ROOT="$(get_project_root)"
    MODEL_DIR=$(get_model_dir "$server" "$model")

    echo_info "项目根目录: $PROJECT_ROOT"
    echo_info "远程模型目录: $MODEL_DIR"
    echo ""

    # 确保远程目录存在
    if ! check_training_dir "$server" "$model"; then
        echo_info "模型目录不存在，创建目录: $MODEL_DIR"
        execute_remote_command "$server" "mkdir -p $MODEL_DIR"
    fi

    echo_info "开始部署 battle_single 到服务器 $server..."

    # ============================================
    # 上传 battle_single 目录
    # ============================================

    if [[ -d "$PROJECT_ROOT/battle_single" ]]; then
        echo_info "上传 battle_single 代码..."
        upload_file_ssh "$server" "$PROJECT_ROOT/battle_single/" "$MODEL_DIR/"
        if [[ $? -ne 0 ]]; then
            echo_error "battle_single 上传失败"
            exit 1
        fi
        echo_success "battle_single 上传完成"
    else
        echo_error "battle_single 目录不存在: $PROJECT_ROOT/battle_single"
        exit 1
    fi

    # ============================================
    # 清理远程 Python 缓存
    # ============================================

    echo_info "清理远程 Python 缓存..."
    execute_remote_command "$server" "find \"$MODEL_DIR/battle_single\" -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; find \"$MODEL_DIR/battle_single\" -type f -name '*.pyc' -delete 2>/dev/null; echo '缓存清理完成'"

    # ============================================
    # 验证上传
    # ============================================

    echo_info ""
    echo_info "验证远程目录结构..."
    echo_info ""
    execute_remote_command "$server" "cd \"$MODEL_DIR\" && ls -la battle_single/ 2>/dev/null || echo 'battle_single 目录内容'"
    echo_info ""

    echo_success "battle_single 部署完成！"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec_deploy_battle "$@"
fi
