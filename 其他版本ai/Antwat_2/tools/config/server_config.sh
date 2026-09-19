#!/bin/bash

# 服务器连接信息

# Server1 - 备用服务器
export SERVER1_HOST="connect.westd.seetacloud.com"
export SERVER1_PORT="37058"
export SERVER1_USER="root"
export SERVER1_PASSWORD="Dy0NFfHGRv30"

# Server2 - 当前使用
export SERVER2_HOST="connect.westc.seetacloud.com"
export SERVER2_PORT="14639"
export SERVER2_USER="root"
export SERVER2_PASSWORD="WxuErkwoJYA4"

# Server3 - 新服务器
export SERVER3_HOST="connect.westb.seetacloud.com"
export SERVER3_PORT="52706"
export SERVER3_USER="root"
export SERVER3_PASSWORD="xClm9ItuLZXv"

# Server4 - 新增服务器
export SERVER4_HOST="connect.westd.seetacloud.com"
export SERVER4_PORT="40768"
export SERVER4_USER="root"
export SERVER4_PASSWORD="R92A1c4/Pnhu"

# Server5 - 新增服务器
export SERVER5_HOST="connect.westb.seetacloud.com"
export SERVER5_PORT="28047"
export SERVER5_USER="root"
export SERVER5_PASSWORD="Elwj9ILwgUp1"

# Server6 - 新增服务器
export SERVER6_HOST="connect.westb.seetacloud.com"
export SERVER6_PORT="35653"
export SERVER6_USER="root"
export SERVER6_PASSWORD="YP34CFaxzcMq"

# Server7 - 新增服务器
export SERVER7_HOST="connect.westd.seetacloud.com"
export SERVER7_PORT="28578"
export SERVER7_USER="root"
export SERVER7_PASSWORD="xsTfmMrJD7DM"

# Server8 - 新增服务器
export SERVER8_HOST="connect.westd.seetacloud.com"
export SERVER8_PORT="15561"
export SERVER8_USER="root"
export SERVER8_PASSWORD="Ouxo0hLqTg4z"

# Server9 - 新增服务器
export SERVER9_HOST="connect.westd.seetacloud.com"
export SERVER9_PORT="26349"
export SERVER9_USER="root"
export SERVER9_PASSWORD="8I0GK1ha1Tk7"

# Server10 - 新增服务器
export SERVER10_HOST="connect.westd.seetacloud.com"
export SERVER10_PORT="19873"
export SERVER10_USER="root"
export SERVER10_PASSWORD="xj/RgSSdUmLC"

# Server11 - 新增服务器
export SERVER11_HOST="connect.westd.seetacloud.com"
export SERVER11_PORT="38081"
export SERVER11_USER="root"
export SERVER11_PASSWORD="yQ58b7Qbtyqv"

# 默认服务器配置
export DEFAULT_SERVER="server2"

# 路径配置
export REMOTE_BASE_PATH="/root/autodl-tmp/AntWar"
export PPO_REMOTE_PATH="${PPO_REMOTE_PATH:-${REMOTE_BASE_PATH}/ppo_v2}"

# 本地路径配置
export LOCAL_BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python命令配置（可被 python_env.sh 覆盖）
export PYTHON_CMD="${PYTHON_CMD:-python}"