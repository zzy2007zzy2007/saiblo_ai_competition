#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> --model <model> deploy_code
# ================================================================================

# 部署代码脚本
# 使用压缩分片传输方式部署 PPO 源码和 requirements.txt
# 流程：本地压缩 -> 分片(200K/片) -> 上传分片 -> 远程合并分片 -> 远程解压 -> 清理

# 加载配置
source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

CHUNK_SIZE="200K"

get_project_root() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    echo "$(cd "${script_dir}/.." && pwd)"
}

exec_deploy_code() {
    local server="$1"
    local model="$2"

    show_usage() {
        echo "用法: $0 <server> <model>"
        echo ""
        echo "功能: 部署模型训练代码到远程服务器"
        echo ""
        echo "传输流程:"
        echo "  本地压缩 -> 分片(${CHUNK_SIZE}/片) -> 上传分片 -> 远程合并分片 -> 远程解压 -> 清理"
    }

    if [[ -z "$server" || -z "$model" ]]; then
        show_usage
        exit 1
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

    local timestamp=$(date +%Y%m%d_%H%M%S)
    local local_archive="/tmp/ppo_deploy_${timestamp}.tar.gz"
    local local_chunk_dir="/tmp/ppo_deploy_${timestamp}_chunks"
    local archive_name=$(basename "$local_archive")
    local remote_archive="$MODEL_DIR/$archive_name"
    local tar_items=""

    # ============================================
    # 步骤 1: 校验本地目录
    # ============================================

    local local_src="${PROJECT_ROOT}/${model}"
    if [[ ! -d "$local_src" ]]; then
        echo_error "模型目录不存在: $local_src"
        exit 1
    fi

    # ============================================
    # 步骤 2: 本地压缩 + 分片
    # ============================================

    echo_info "步骤 1/6: 本地压缩 ${model}..."
    cd "$PROJECT_ROOT"
    tar -czf "$local_archive" -C "${model}" .
    if [[ $? -ne 0 ]]; then
        echo_error "本地压缩失败"
        rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    local archive_size=$(du -h "$local_archive" | cut -f1)
    echo_success "压缩完成，压缩包大小: $archive_size"

    echo_info "分片压缩包（${CHUNK_SIZE}/片）..."
    mkdir -p "$local_chunk_dir"
    split -b "$CHUNK_SIZE" -d "$local_archive" "${local_chunk_dir}/chunk_"
    if [[ $? -ne 0 ]]; then
        echo_error "分片失败"
        rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    local chunk_count=$(ls -1 "$local_chunk_dir" | wc -l | tr -d ' ')
    echo_success "分片完成，共 ${chunk_count} 个分片"
    echo ""

    # ============================================
    # 步骤 3: 确保远程目录存在
    # ============================================

    local remote_chunk_dir="$MODEL_DIR/.ppo_chunks_${timestamp}"

    echo_info "步骤 2/6: 确保远程目录存在..."
    execute_remote_command "$server" "mkdir -p '$MODEL_DIR'" || {
        echo_error "无法创建远程目录: $MODEL_DIR"
        rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    }
    execute_remote_command "$server" "mkdir -p '$remote_chunk_dir'"
    echo_success "远程目录准备完成"
    echo ""

    # ============================================
    # 步骤 4: 上传所有分片
    # ============================================

    echo_info "步骤 3/6: 上传分片到服务器（共 ${chunk_count} 个）..."
    local uploaded=0
    local failed=0

    for chunk_file in "$local_chunk_dir"/chunk_*; do
        local chunk_name=$(basename "$chunk_file")
        upload_file_ssh "$server" "$chunk_file" "${remote_chunk_dir}/"
        if [[ $? -ne 0 ]]; then
            echo_error "分片上传失败: $chunk_name"
            ((failed++))
        else
            ((uploaded++))
        fi
    done

    if [[ $failed -gt 0 ]]; then
        echo_error "上传失败数: $failed"
        execute_remote_command "$server" "rm -rf '$remote_chunk_dir'"
        rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    fi

    echo_success "所有分片上传完成（${uploaded}/${chunk_count}）"
    echo ""

    # ============================================
    # 步骤 5: 远程合并分片
    # ============================================

    echo_info "步骤 4/6: 远程服务器合并分片..."
    execute_remote_command "$server" "cd '$remote_chunk_dir' && cat chunk_* > '$remote_archive'" || {
        echo_error "远程合并分片失败"
        execute_remote_command "$server" "rm -rf '$remote_chunk_dir'" 2>/dev/null
        rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    }
    echo_success "分片合并完成"
    echo ""

    # ============================================
    # 步骤 6: 远程解压
    # ============================================

    echo_info "步骤 5/6: 远程解压..."
    execute_remote_command "$server" "cd '$MODEL_DIR' && tar -xzf '$remote_archive'" || {
        echo_error "远程解压失败"
        execute_remote_command "$server" "rm -f '$remote_archive' && rm -rf '$remote_chunk_dir'"
        rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null
        exit 1
    }
    echo_success "远程解压完成"

    # ============================================
    # 步骤 7: 清理临时文件 + 清理 Python 缓存
    # ============================================

    echo_info "步骤 6/6: 清理..."
    execute_remote_command "$server" "rm -f '$remote_archive' && rm -rf '$remote_chunk_dir'"
    rm -rf "$local_archive" "$local_chunk_dir" 2>/dev/null

    execute_remote_command "$server" "find '$MODEL_DIR' -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; find '$MODEL_DIR' -type f -name '*.pyc' -delete 2>/dev/null"
    echo_success "清理完成"
    echo ""

    # ============================================
    # 验证部署结果
    # ============================================

    echo_info "验证远程目录结构..."
    echo ""
    execute_remote_command "$server" "cd '$MODEL_DIR' && ls -la 2>/dev/null | head -20"
    echo ""

    echo_success "代码部署完成！"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    exec_deploy_code "$@"
fi
