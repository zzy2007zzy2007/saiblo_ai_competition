#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> deploy_sdk
# ================================================================================

# 部署 SDK 脚本
# 使用压缩分片传输方式部署 Ant-Game 和 baselines
# 流程：本地压缩 -> 分片(8M/片) -> 上传分片 -> 远程合并分片 -> 远程解压 -> 清理

# 加载配置（ssh_helper.sh 已包含 config_helper.sh）
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

CHUNK_SIZE="10M"

# 获取项目根目录
get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "$(cd "${script_dir}/.." && pwd)"
}

exec_deploy_sdk() {
    local server="$1"
    local model="$2"

    show_usage() {
        echo "用法: $0 <server> <model>"
        echo ""
        echo "功能: 使用压缩分片传输方式部署 Ant-Game 和 baselines 到远程服务器"
        echo ""
        echo "部署内容:"
        echo "  1. Ant-Game SDK 目录"
        echo "  2. baselines 目录"
        echo ""
        echo "传输流程:"
        echo "  本地压缩 -> 分片(${CHUNK_SIZE}/片) -> 上传分片 -> 远程合并 -> 远程解压 -> 清理"
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

    echo_info "开始部署 SDK 到服务器 $server..."

    # ============================================
    # 步骤 1/7: 检查本地目录是否存在
    # ============================================

    if [[ ! -d "$PROJECT_ROOT/Ant-Game" ]]; then
        echo_error "Ant-Game 目录不存在: $PROJECT_ROOT/Ant-Game"
        exit 1
    fi

    if [[ ! -d "$PROJECT_ROOT/baselines" ]]; then
        echo_error "baselines 目录不存在: $PROJECT_ROOT/baselines"
        exit 1
    fi

    # ============================================
    # 步骤 2/7: 本地压缩 + 分片
    # ============================================

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local local_sdk_archive="/tmp/sdk_deploy_${timestamp}.tar.gz"
    local local_chunk_dir="/tmp/sdk_deploy_${timestamp}_chunks"

    rm -f "$local_sdk_archive" 2>/dev/null
    rm -rf "$local_chunk_dir" 2>/dev/null

    echo_info "步骤 2/7: 本地压缩 Ant-Game 和 baselines..."
    tar -C "$PROJECT_ROOT" -czf "$local_sdk_archive" Ant-Game baselines
    if [[ $? -ne 0 ]]; then
        echo_error "本地压缩失败"
        rm -rf "$local_sdk_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    local archive_size=$(du -h "$local_sdk_archive" | cut -f1)
    echo_success "压缩完成，压缩包大小: $archive_size"

    echo_info "分片压缩包（${CHUNK_SIZE}/片）..."
    mkdir -p "$local_chunk_dir"
    split -b "$CHUNK_SIZE" -d "$local_sdk_archive" "${local_chunk_dir}/chunk_"
    if [[ $? -ne 0 ]]; then
        echo_error "分片失败"
        rm -rf "$local_sdk_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    local chunk_count=$(ls -1 "$local_chunk_dir" | wc -l | tr -d ' ')
    echo_success "分片完成，共 ${chunk_count} 个分片"
    echo ""

    # ============================================
    # 步骤 3/7: 确保远程目录存在
    # ============================================

    echo_info "步骤 3/7: 确保远程目录存在..."
    if ! check_training_dir "$server" "$model"; then
        echo_info "模型目录不存在，创建目录: $MODEL_DIR"
        execute_remote_command "$server" "mkdir -p '$MODEL_DIR'"
    fi
    local remote_chunk_dir="$MODEL_BASE_DIR/.sdk_chunks_${timestamp}"
    execute_remote_command "$server" "mkdir -p '$remote_chunk_dir'"

    # 验证远程分片目录已创建
    execute_remote_command "$server" "test -d '$remote_chunk_dir' && echo 'OK'" || {
        echo_error "远程分片目录不存在: $remote_chunk_dir"
        exit 1
    }
    echo_success "远程目录准备完成"
    echo ""

    # ============================================
    # 步骤 4/7: 上传所有分片
    # ============================================

    echo_info "步骤 4/7: 上传分片到服务器（共 ${chunk_count} 个）..."
    echo_info "  本地分片目录: $local_chunk_dir"
    echo_info "  远程分片目录: $remote_chunk_dir"
    echo_info "  服务器: $server"
    echo_info "  分片大小: ${CHUNK_SIZE}"
    local uploaded=0
    local failed=0

    for chunk_file in "$local_chunk_dir"/chunk_*; do
        local chunk_name=$(basename "$chunk_file")
        local chunk_size=$(du -h "$chunk_file" | cut -f1)
        echo_info "[${chunk_name}/${chunk_count}] 上传中 (${chunk_size}) -> ${remote_chunk_dir}/ ..."
        upload_file_ssh "$server" "$chunk_file" "${remote_chunk_dir}/"
        if [[ $? -ne 0 ]]; then
            echo_error "分片上传失败: $chunk_name (${server}:${remote_chunk_dir}/)"
            ((failed++))
        else
            echo_success "[${chunk_name}] 上传完成"
            ((uploaded++))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        echo_error "上传失败数: $failed"
        execute_remote_command "$server" "rm -rf '$remote_chunk_dir'"
        rm -rf "$local_sdk_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    echo_success "所有分片上传完成（${uploaded}/${chunk_count}）"
    echo ""

    # ============================================
    # 步骤 5/7: 远程合并分片
    # ============================================

    local remote_archive="$MODEL_BASE_DIR/sdk_deploy_${timestamp}.tar.gz"

    echo_info "步骤 5/7: 远程服务器合并分片..."
    execute_remote_command "$server" "cd '$remote_chunk_dir' && cat chunk_* > '$remote_archive'"
    if [[ $? -ne 0 ]]; then
        echo_error "远程合并分片失败"
        execute_remote_command "$server" "rm -rf '$remote_chunk_dir'" 2>/dev/null
        rm -rf "$local_sdk_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi
    echo_success "分片合并完成"
    echo ""

    # ============================================
    # 步骤 6/7: 远程解压（先解压到临时目录，成功后再替换）
    # ============================================

    echo_info "步骤 6/7: 远程服务器解压..."

    local remote_staging="$MODEL_BASE_DIR/.sdk_staging_${timestamp}"
    execute_remote_command "$server" "mkdir -p '$remote_staging'"

    # 解压到临时目录
    execute_remote_command "$server" "tar -C '$remote_staging' -xzf '$remote_archive'"
    if [[ $? -ne 0 ]]; then
        echo_error "远程解压失败"
        execute_remote_command "$server" "rm -f '$remote_archive' && rm -rf '$remote_chunk_dir' && rm -rf '$remote_staging'"
        rm -rf "$local_sdk_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    # 解压成功，替换旧目录
    execute_remote_command "$server" "cd '$MODEL_BASE_DIR' && rm -rf Ant-Game baselines && mv '$remote_staging/Ant-Game' . && mv '$remote_staging/baselines' . && rm -rf '$remote_staging'"
    if [[ $? -ne 0 ]]; then
        echo_error "替换旧 SDK 失败"
        execute_remote_command "$server" "rm -rf '$remote_staging'" 2>/dev/null
        exit 1
    fi

    echo_success "远程解压完成"
    echo ""

    # ============================================
    # 步骤 7/7: 清理远程临时文件
    # ============================================

    echo_info "步骤 7/7: 清理临时文件..."
    execute_remote_command "$server" "rm -f '$remote_archive' && rm -rf '$remote_chunk_dir'"
    echo_success "远程临时文件已清理"

    rm -rf "$local_sdk_archive" "$local_chunk_dir" 2>/dev/null
    echo_success "本地临时文件已清理"
    echo ""

    # ============================================
    # 验证部署结果
    # ============================================

    echo_info "验证远程目录结构..."
    echo ""
    execute_remote_command "$server" "ls -la '$MODEL_BASE_DIR'"
    echo ""

    echo_success "SDK 部署完成！"
    echo ""
    echo_info "部署摘要:"
    echo "  服务器:     $server"
    echo "  模型:       $model"
    echo "  目标目录:   $MODEL_DIR"
    echo "  分片大小:   ${CHUNK_SIZE}"
    echo "  分片数量:   ${chunk_count}"
    echo "  已部署:"
    echo "    - Ant-Game SDK"
    echo "    - baselines"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec_deploy_sdk "$@"
fi
