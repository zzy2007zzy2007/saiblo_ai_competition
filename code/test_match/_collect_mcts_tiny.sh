#!/usr/bin/env bash
# MCTS 教师采集：4 进程 × 1 局（少并行，先保证能出数据）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=mcts PRIOR_MCTS_ITERS=64 PRIOR_MCTS_K=24 PRIOR_MAX_ROUNDS=200
echo "############ MCTS 教师采集（4 进程 × 1 局，64 iter/200 回合）  $(date +%H:%M:%S)"
for w in 0 1 2 3; do
  ( seed=$((9800 + w)); $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -3 ) &
done
wait
echo "完成：$(ls -1 training_history/prior_labels/m*.npz 2>/dev/null | wc -l) 个文件  $(date +%H:%M:%S)"