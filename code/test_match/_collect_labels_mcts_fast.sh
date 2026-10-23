#!/usr/bin/env bash
# MCTS 访问平均教师采集（低成本档：iterations=64, max_rounds=240）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=mcts
export PRIOR_MCTS_ITERS=64
export PRIOR_MCTS_K=24
export PRIOR_MAX_ROUNDS=240
echo "############ MCTS 教师采集（64 iter / 240 回合）：24 局（8 并行 × 3 局）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7; do
  (
    for j in 0 1 2; do
      seed=$((9700 + w * 10 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -2
    done
  ) &
done
wait
echo ""
echo "MCTS 采集完成：$(ls -1 training_history/prior_labels/m*.npz 2>/dev/null | wc -l) 个文件  $(date +%H:%M:%S)"