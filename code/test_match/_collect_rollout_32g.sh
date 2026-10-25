#!/usr/bin/env bash
# 补齐 rollout 教师数据：32 局（4 进程 × 8 局），种子 g9200-g9231
# 与 f10 的教师同一定义：posnet_A 自对弈、全枚举候选(k=300)、h=16 rollout、值网打分
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=rollout PRIOR_MAX_ROUNDS=200
echo "############ rollout 教师采集 32 局（4 进程 × 8 局）  $(date +%H:%M:%S)"
for w in 0 1 2 3; do
  (
    for j in 0 1 2 3 4 5 6 7; do
      seed=$((9200 + w * 8 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -1
    done
  ) &
done
wait
echo ""
echo "完成：g*.npz 共 $(ls -1 training_history/prior_labels/g*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"