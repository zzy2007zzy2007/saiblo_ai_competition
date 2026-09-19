#!/bin/bash

# 配置帮助函数

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

# 服务器配置
SERVERS=("server1" "server2" "server3" "server4" "server5" "server6" "server7" "server8" "server9" "server10" "server11")

# 服务器 IP 映射（需要根据实际情况修改）
SERVER_IPS=("server1.example.com" "server2.example.com" "server3.example.com" "server4.example.com" "server5.example.com" "server6.example.com" "server7.example.com")

# 模型目录基础路径
MODEL_BASE_DIR="/root/autodl-tmp/AntWar"

# 获取服务器 IP
get_server_ip() {
    local server_name="$1"
    for i in "${!SERVERS[@]}"; do
        if [[ "${SERVERS[$i]}" == "$server_name" ]]; then
            echo "${SERVER_IPS[$i]}"
            return 0
        fi
    done
    return 1
}

# 验证服务器名称
validate_server() {
    local server_name="$1"
    for server in "${SERVERS[@]}"; do
        if [[ "$server" == "$server_name" ]]; then
            return 0
        fi
    done
    return 1
}

# 获取模型目录
get_model_dir() {
    local server_name="$1"
    local model_name="$2"
    echo "${MODEL_BASE_DIR}/${model_name}"
}

# 检查训练目录是否存在
check_training_dir() {
    local server_name="$1"
    local model_name="$2"
    local model_dir="${MODEL_BASE_DIR}/${model_name}"
    execute_remote_command "$server_name" "[ -d '$model_dir' ]"
    return $?
}

# 显示错误信息
echo_error() {
    echo -e "\033[31m错误: $1\033[0m"
}

# 显示成功信息
echo_success() {
    echo -e "\033[32m成功: $1\033[0m"
}

# 显示信息
echo_info() {
    echo -e "\033[34m信息: $1\033[0m"
}

# 显示警告信息
echo_warning() {
    echo -e "\033[33m警告: $1\033[0m"
}
