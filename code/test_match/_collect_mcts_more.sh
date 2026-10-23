#!/usr/bin/env bash
# MCTS 教师扩数据：4 进程 × 3 局 = 12 局（单局 8.2 min ⇒ 约 25-30 min）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=mcts PRIOR_MCTS_ITERS=64 PRIOR_MCTS_K=24 PRIOR_MAX_ROUNDS=200
echo "############ MCTS 教师扩数据：12 局（4 进程 × 3 局）  $(date +%H:%M:%S)"
for w in 0 1 2 3; do
  (
    for j in 0 1 2; do
      seed=$((9810 + w * 10 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -2
    done
  ) &
done
wait
echo "完成：m*.npz 共 $(ls -1 training_history/prior_labels/m*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"