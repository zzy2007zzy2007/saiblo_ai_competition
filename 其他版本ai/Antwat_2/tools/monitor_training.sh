#!/bin/bash
# PPO 训练监控脚本 —— 一次性查看 6 项训练指标
# 用法: bash tools/monitor_training.sh

set -e

SERVER="server9"
MODEL="ppo_v2"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MANAGER="bash ${SCRIPT_DIR}/manager.sh --server ${SERVER} --model ${MODEL}"

echo "=========================================="
echo "  查找最新日志目录..."
echo "=========================================="
$MANAGER exec "ls -td /root/autodl-tmp/AntWar/outputs/20*/ 2>/dev/null | head -1"
echo ""

echo "=========================================="
echo "  1/6  SelfPlay 技术参数表"
echo "=========================================="
$MANAGER train_metrics_table
echo ""
echo ""

echo "=========================================="
echo "  2/6  动作执行次数统计"
echo "=========================================="
$MANAGER train_action_table
echo ""
echo ""

echo "=========================================="
echo "  3/6  Action Reward 原始数值"
echo "=========================================="
$MANAGER train_action_reward_val_table
echo ""
echo ""

echo "=========================================="
echo "  4/6  Action Reward 占比统计"
echo "=========================================="
$MANAGER train_action_reward_table
echo ""
echo ""

echo "=========================================="
echo "  5/6  Baseline 对战横表"
echo "=========================================="
$MANAGER train_battle_table
echo ""
echo ""

echo "=========================================="
echo "  6/6  异常扫描"
echo "=========================================="
$MANAGER scan_anomalies
echo ""
