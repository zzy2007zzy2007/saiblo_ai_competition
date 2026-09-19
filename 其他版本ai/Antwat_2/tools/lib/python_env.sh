#!/bin/bash

# Python 环境统一配置文件
# 所有脚本应引用此文件获取 Python 路径和 conda 环境配置

# 防止重复加载
if [ -n "_PYTHON_ENV_LOADED" ]; then
    return 0
fi
_PYTHON_ENV_LOADED=1

# ============================================
# Python 路径配置
# ============================================

# 优先使用 conda 环境中的 Python
CONDA_PYTHON="/root/miniconda3/bin/python"
CONDA_PYTHON3="/root/miniconda3/bin/python3"

# 系统 Python 备选
SYSTEM_PYTHON="/usr/bin/python"
SYSTEM_PYTHON3="/usr/bin/python3"

# 统一 Python 命令 - 优先使用 conda 版本
if [ -x "$CONDA_PYTHON" ]; then
    export PYTHON_CMD="$CONDA_PYTHON"
elif [ -x "$CONDA_PYTHON3" ]; then
    export PYTHON_CMD="$CONDA_PYTHON3"
elif [ -x "$SYSTEM_PYTHON" ]; then
    export PYTHON_CMD="$SYSTEM_PYTHON"
elif [ -x "$SYSTEM_PYTHON3" ]; then
    export PYTHON_CMD="$SYSTEM_PYTHON3"
else
    export PYTHON_CMD="python3"
fi

# ============================================
# 路径环境配置
# ============================================

# 基础 PATH（包含 conda 和系统命令）
export BASE_PATH="/root/miniconda3/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin"

# 设置 PATH（保留系统路径）
export PATH="$BASE_PATH:$PATH"

# ============================================
# 辅助函数
# ============================================

# 检查 Python 是否可用
check_python() {
    if "$PYTHON_CMD" -c "import sys; sys.exit(0)" 2>/dev/null; then
        echo "Python 可用: $($PYTHON_CMD --version)"
        return 0
    else
        echo "Python 不可用"
        return 1
    fi
}

# 获取 Python 版本
get_python_version() {
    "$PYTHON_CMD" --version 2>/dev/null || echo "未知"
}

# 执行 Python 命令（带环境变量）
run_python() {
    local cmd="$1"
    shift
    PYTHONPATH="$PYTHONPATH" "$PYTHON_CMD" -c "$cmd" "$@"
}