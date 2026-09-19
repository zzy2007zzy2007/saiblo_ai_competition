#!/bin/bash
# ============================================================
# WARNING: 不要在本地测试！请在 Server5 的 /tmp/battle 文件夹测试！
# WARNING: Do NOT test locally! Please test in /tmp/battle on Server5!
# ============================================================

set -e

BATTLE_DIR="/tmp/battle_single"
mkdir -p "$BATTLE_DIR/logs"
mkdir -p "$BATTLE_DIR/results"

cd "$BATTLE_DIR"

if [ -n "$1" ] && [ -n "$2" ]; then
    AGENT1="$1"
    AGENT2="$2"
else
    AGENT1="ann_v1"
    AGENT2="gen99"
fi

EPISODES=${3:-10}
WORKERS=${4:-4}

echo "=== Battle Single 对战评测 ==="
echo "对战双方: $AGENT1 vs $AGENT2"
echo "对战局数: $EPISODES"
echo "并行线程: $WORKERS"
echo "==============================="

python -m battle_single.src.battle_single_cli \
    --agent1 "$AGENT1" \
    --agent2 "$AGENT2" \
    --episodes "$EPISODES" \
    --workers "$WORKERS" \
    --output "$BATTLE_DIR/results/report.txt"

echo ""
echo "=== 评测完成 ==="
echo "报告位置: $BATTLE_DIR/results/report.txt"