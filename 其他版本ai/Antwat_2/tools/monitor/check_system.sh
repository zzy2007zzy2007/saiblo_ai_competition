#!/bin/bash

# ⚠️  重要提示  ⚠️
# 请不要直接调用此脚本！
# 请通过 tools/manager.sh 统一调用，例如：
#   bash tools/manager.sh --server <server> monitor_system
# ================================================================================

# ================================================================================
# 系统资源监控脚本
# Usage: bash tools/monitor/check_system.sh <server> [--watch]
# ================================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../lib/config_helper.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../lib/ssh_helper.sh"

SERVER="$1"
WATCH_MODE="$2"

show_usage() {
    echo "用法: $0 <server> [--watch]"
    echo ""
    echo "功能: 监控系统资源使用情况"
    echo ""
    echo "选项:"
    echo "  --watch    持续监控模式（每秒刷新）"
    echo ""
    echo "监控指标:"
    echo "  - CPU 使用率"
    echo "  - 内存使用情况"
    echo "  - 磁盘使用情况"
    echo "  - 网络流量"
    echo "  - GPU 状态（如可用）"
    echo ""
    echo "示例:"
    echo "  $0 server1"
    echo "  $0 server1 --watch"
    exit 1
}

if [[ -z "$SERVER" ]]; then
    show_usage
fi

check_server_connection "$SERVER"
if [[ $? -ne 0 ]]; then
    exit 1
fi

get_system_stats() {
    local server="$1"
    local command="python3 << 'PYEOF'
import psutil
import json
from datetime import datetime

def get_stats():
    try:
        net_io = psutil.net_io_counters()
        stats = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_percent': psutil.cpu_percent(interval=0.5),
            'cpu_count': psutil.cpu_count(),
            'memory': {
                'total': round(psutil.virtual_memory().total / (1024**3), 2),
                'available': round(psutil.virtual_memory().available / (1024**3), 2),
                'used': round(psutil.virtual_memory().used / (1024**3), 2),
                'percent': psutil.virtual_memory().percent
            },
            'swap': {
                'total': round(psutil.swap_memory().total / (1024**3), 2),
                'used': round(psutil.swap_memory().used / (1024**3), 2),
                'percent': psutil.swap_memory().percent
            },
            'disk': {
                'total': round(psutil.disk_usage('/').total / (1024**3), 2),
                'used': round(psutil.disk_usage('/').used / (1024**3), 2),
                'free': round(psutil.disk_usage('/').free / (1024**3), 2),
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': round(net_io.bytes_sent / (1024**2), 2),
                'bytes_recv': round(net_io.bytes_recv / (1024**2), 2),
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            },
            'processes': len(psutil.pids()),
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        return stats
    except Exception as e:
        return {'error': str(e)}

print(json.dumps(get_stats()))
PYEOF"
    execute_remote_command "$server" "$command"
}

get_gpu_stats() {
    local server="$1"
    local command="nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo ''"
    execute_remote_command "$server" "$command"
}

print_stats() {
    local server="$1"

    echo "=========================================="
    echo "        系统资源监控 - $server"
    echo "=========================================="

    local stats=$(get_system_stats "$server")

    if echo "$stats" | grep -q '"error"'; then
        echo_error "获取系统信息失败"
        return
    fi

    local timestamp=$(echo "$stats" | python3 -c "import sys,json; print(json.load(sys.stdin)['timestamp'])")
    local cpu=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['cpu_percent'])")
    local cpu_count=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['cpu_count'])")
    local load1=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['load_avg'][0])")
    local load5=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['load_avg'][1])")

    echo "时间: $timestamp"
    echo ""
    echo "【CPU 信息】"
    echo "  CPU 使用率:     ${cpu}%"
    echo "  CPU 核心数:     $cpu_count"
    echo "  系统负载:       ${load1}, ${load5} (1m, 5m)"
    echo ""

    local mem_total=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['memory']['total'])")
    local mem_used=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['memory']['used'])")
    local mem_percent=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['memory']['percent'])")
    local swap_total=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['swap']['total'])")
    local swap_used=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['swap']['used'])")
    local swap_percent=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['swap']['percent'])")

    echo "【内存信息】"
    echo "  物理内存:       ${mem_used} GB / ${mem_total} GB (${mem_percent}%)"
    echo "  Swap:           ${swap_used} GB / ${swap_total} GB (${swap_percent}%)"
    echo ""

    local disk_total=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['disk']['total'])")
    local disk_used=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['disk']['used'])")
    local disk_percent=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['disk']['percent'])")

    echo "【磁盘信息】"
    echo "  磁盘使用:       ${disk_used} GB / ${disk_total} GB (${disk_percent}%)"
    echo ""

    local net_sent=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['network']['bytes_sent'])")
    local net_recv=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['network']['bytes_recv'])")

    echo "【网络流量】（累计）"
    echo "  已发送:         ${net_sent} MB"
    echo "  已接收:         ${net_recv} MB"
    echo ""

    local processes=$(echo "$stats" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['processes'])")
    echo "【进程信息】"
    echo "  总进程数:       $processes"
    echo ""

    echo "【GPU 信息】（如有）"
    local gpu_info=$(get_gpu_stats "$server")
    if [[ -n "$gpu_info" ]]; then
        echo "  ID | Name               | GPU% | Memory Used | Memory Total | Temp"
        echo "  $(echo "$gpu_info" | head -n 4 | while read line; do echo "$line" | sed 's/,/ | /g'; done)"
    else
        echo "  GPU 信息不可用"
    fi

    echo "=========================================="
}

exec_check_system() {
    local server="$1"
    local watch_mode="$2"

    show_usage() {
        echo "用法: $0 <server> [--watch]"
        echo ""
        echo "功能: 监控系统资源使用情况"
        echo ""
        echo "选项:"
        echo "  --watch    持续监控模式（每秒刷新）"
        echo ""
        echo "监控指标:"
        echo "  - CPU 使用率"
        echo "  - 内存使用情况"
        echo "  - 磁盘使用情况"
        echo "  - 网络流量"
        echo "  - GPU 状态（如可用）"
        echo ""
        echo "示例:"
        echo "  $0 server1"
        echo "  $0 server1 --watch"
        exit 1
    }

    if [[ -z "$server" ]]; then
        show_usage
    fi

    check_server_connection "$server"
    if [[ $? -ne 0 ]]; then
        exit 1
    fi

    SERVER="$server"
    WATCH_MODE="$watch_mode"

    if [[ "$WATCH_MODE" == "--watch" ]]; then
        while true; do
            clear
            print_stats "$server"
            sleep 2
        done
    else
        print_stats "$server"
    fi
}

exec_check_system "$@"