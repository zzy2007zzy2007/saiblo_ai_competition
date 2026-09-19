#!/bin/bash

# SSH 帮助函数

# 始终使用 BASH_SOURCE[0] 计算正确的路径
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_TOOLS_DIR="$(cd "${_LIB_DIR}/.." && pwd)"

# 如果未被设置（从 manager.sh 直接调用），则设置全局变量
if [[ -z "$LIB_DIR" ]]; then
    LIB_DIR="$_LIB_DIR"
fi
if [[ -z "$TOOLS_DIR" ]]; then
    TOOLS_DIR="$_TOOLS_DIR"
fi

# 加载配置
source "${_LIB_DIR}/config_helper.sh"
source "${_TOOLS_DIR}/config/server_config.sh"

# 检查服务器连接
check_server_connection() {
    local server_name="$1"
    if [[ -z "$server_name" ]]; then
        echo_error "服务器名称为空"
        return 1
    fi

    set_server "$server_name"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    if ! sshpass -p "${SERVER_PASSWORD}" ssh -p "${SERVER_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 "${SERVER_USER}@${SERVER_HOST}" "echo connected" &>/dev/null; then
        echo_error "无法连接到服务器: $server_name ($SERVER_USER@${SERVER_HOST}:${SERVER_PORT})"
        return 1
    fi
    return 0
}

# 设置服务器信息
set_server() {
    local server_name="$1"
    case "$server_name" in
        "server1")
            export SERVER_HOST="${SERVER1_HOST}"
            export SERVER_PORT="${SERVER1_PORT}"
            export SERVER_USER="${SERVER1_USER}"
            export SERVER_PASSWORD="${SERVER1_PASSWORD}"
            ;;
        "server2")
            export SERVER_HOST="${SERVER2_HOST}"
            export SERVER_PORT="${SERVER2_PORT}"
            export SERVER_USER="${SERVER2_USER}"
            export SERVER_PASSWORD="${SERVER2_PASSWORD}"
            ;;
        "server3")
            export SERVER_HOST="${SERVER3_HOST}"
            export SERVER_PORT="${SERVER3_PORT}"
            export SERVER_USER="${SERVER3_USER}"
            export SERVER_PASSWORD="${SERVER3_PASSWORD}"
            ;;
        "server4")
            export SERVER_HOST="${SERVER4_HOST}"
            export SERVER_PORT="${SERVER4_PORT}"
            export SERVER_USER="${SERVER4_USER}"
            export SERVER_PASSWORD="${SERVER4_PASSWORD}"
            ;;
        "server5")
            export SERVER_HOST="${SERVER5_HOST}"
            export SERVER_PORT="${SERVER5_PORT}"
            export SERVER_USER="${SERVER5_USER}"
            export SERVER_PASSWORD="${SERVER5_PASSWORD}"
            ;;
        "server6")
            export SERVER_HOST="${SERVER6_HOST}"
            export SERVER_PORT="${SERVER6_PORT}"
            export SERVER_USER="${SERVER6_USER}"
            export SERVER_PASSWORD="${SERVER6_PASSWORD}"
            ;;
        "server7")
            export SERVER_HOST="${SERVER7_HOST}"
            export SERVER_PORT="${SERVER7_PORT}"
            export SERVER_USER="${SERVER7_USER}"
            export SERVER_PASSWORD="${SERVER7_PASSWORD}"
            ;;
        "server8")
            export SERVER_HOST="${SERVER8_HOST}"
            export SERVER_PORT="${SERVER8_PORT}"
            export SERVER_USER="${SERVER8_USER}"
            export SERVER_PASSWORD="${SERVER8_PASSWORD}"
            ;;
        "server9")
            export SERVER_HOST="${SERVER9_HOST}"
            export SERVER_PORT="${SERVER9_PORT}"
            export SERVER_USER="${SERVER9_USER}"
            export SERVER_PASSWORD="${SERVER9_PASSWORD}"
            ;;
        "server10")
            export SERVER_HOST="${SERVER10_HOST}"
            export SERVER_PORT="${SERVER10_PORT}"
            export SERVER_USER="${SERVER10_USER}"
            export SERVER_PASSWORD="${SERVER10_PASSWORD}"
            ;;
        "server11")
            export SERVER_HOST="${SERVER11_HOST}"
            export SERVER_PORT="${SERVER11_PORT}"
            export SERVER_USER="${SERVER11_USER}"
            export SERVER_PASSWORD="${SERVER11_PASSWORD}"
            ;;
        *)
            echo_error "未知的服务器: $server_name"
            return 1
            ;;
    esac
    return 0
}

# 执行远程命令
execute_remote_command() {
    local server_name="$1"
    local command="$2"

    set_server "$server_name"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    local python_path="/root/miniconda3/bin/python"
    local python3_path="/root/miniconda3/bin/python3"
    local path_setup="export PATH=\"/root/miniconda3/bin:\$PATH\" && source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true"

    sshpass -p "${SERVER_PASSWORD}" ssh -p "${SERVER_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${SERVER_USER}@${SERVER_HOST}" "cd /root/autodl-tmp/AntWar && $path_setup && $command"
    return $?
}

# 上传文件
upload_file_ssh() {
    local server_name="$1"
    local local_path="$2"
    local remote_path="$3"

    set_server "$server_name"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    if [[ ! -f "$local_path" && ! -d "$local_path" ]]; then
        echo_error "本地文件或目录不存在: $local_path"
        return 1
    fi

    sshpass -p "${SERVER_PASSWORD}" scp -P "${SERVER_PORT}" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=60 \
        -o ServerAliveInterval=10 \
        -r "$local_path" "${SERVER_USER}@${SERVER_HOST}:${remote_path}"
    return $?
}

# 下载文件
download_file_ssh() {
    local server_name="$1"
    local remote_path="$2"
    local local_path="$3"

    set_server "$server_name"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    sshpass -p "${SERVER_PASSWORD}" scp -P "${SERVER_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${SERVER_USER}@${SERVER_HOST}:${remote_path}" "$local_path"
    return $?
}

# 在远程服务器上执行命令并返回输出（单次执行，无session）
exec_remote_no_screen() {
    local server_name="$1"
    local command="$2"

    set_server "$server_name"
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    local python_path="/root/miniconda3/bin/python"
    local python3_path="/root/miniconda3/bin/python3"
    local path_setup="export PATH=\"/root/miniconda3/bin:\$PATH\" && source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true"

    sshpass -p "${SERVER_PASSWORD}" ssh -p "${SERVER_PORT}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "${SERVER_USER}@${SERVER_HOST}" "cd /root/autodl-tmp/AntWar && $path_setup && $command"
    return $?
}
