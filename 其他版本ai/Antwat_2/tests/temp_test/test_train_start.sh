#!/bin/bash
# 测试训练启动

cd /root/autodl-tmp/AntWar/ppo_v1

export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/AntWar/ppo_v1/ppo/src:/root/autodl-tmp/AntWar/ppo_v1/Ant-Game

echo "=== 测试启动训练 ==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "Python路径: $(which python3)"
echo "Python版本: $(python3 --version)"

# 创建新的训练目录
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="./logs/${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "训练日志目录: $LOG_DIR"

# 启动训练
echo "=== 启动训练 ==="
screen -dmS ppo_v1_training bash -c "cd /root/autodl-tmp/AntWar/ppo_v1 && \
export PATH=/root/miniconda3/bin:\$PATH && \
export PYTHONPATH=/root/autodl-tmp/AntWar/ppo_v1/ppo/src:/root/autodl-tmp/AntWar/ppo_v1/Ant-Game && \
python3 ppo/train.py \
  --episodes 101 \
  --n-envs 2 \
  --battle-interval 10 \
  --n-battles 2 \
  2>&1 | tee ${LOG_DIR}/train.log"

sleep 5

echo "=== 检查训练状态 ==="
screen -ls
ps aux | grep train | grep -v grep

echo "=== 检查日志目录 ==="
ls -la $LOG_DIR/
