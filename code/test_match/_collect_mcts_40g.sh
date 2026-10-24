#!/usr/bin/env bash
# Goal 步骤(5)：补 40 局 MCTS 教师采集（4 并行 × 10 局，实测 8.2 min/局 ⇒ 约 90 分钟）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=mcts PRIOR_MCTS_ITERS=64 PRIOR_MCTS_K=24 PRIOR_MAX_ROUNDS=200
echo "############ MCTS 教师 40 局采集（4 进程 × 10 局）  $(date +%H:%M:%S)"
for w in 0 1 2 3; do
  (
    for j in 0 1 2 3 4 5 6 7 8 9; do
      seed=$((9900 + w * 10 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -1
    done
  ) &
done
wait
echo ""
echo "完成：m*.npz 共 $(ls -1 training_history/prior_labels/m*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"