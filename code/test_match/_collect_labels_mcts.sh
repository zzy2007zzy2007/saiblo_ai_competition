#!/usr/bin/env bash
# MCTS 访问平均教师采集（2026-10-23 夜）
# 值网 rollout 排序教师已被证伪（Spearman +0.428 而棋力 50%），改用 BundleMCTS 的 visit_policy。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=mcts
export PRIOR_MCTS_ITERS=256
export PRIOR_MCTS_K=24
mkdir -p training_history/prior_labels_mcts
echo "############ MCTS 教师采集：40 局（8 并行 × 5 局）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7; do
  (
    for j in 0 1 2 3 4; do
      seed=$((9600 + w * 10 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -2
    done
  ) &
done
wait
echo ""
echo "MCTS 采集完成：$(ls -1 training_history/prior_labels/*.npz 2>/dev/null | wc -l) 个文件  $(date +%H:%M:%S)"