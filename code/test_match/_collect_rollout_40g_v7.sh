#!/usr/bin/env bash
# 再采 40 局 rollout 教师（种子 g9340-g9379），10 并行
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=rollout PRIOR_MAX_ROUNDS=200 PRIOR_TEACHER_EVERY=1
export AZAI_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "############ rollout 教师第二批 40 局（g9340-g9379）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7 8 9; do
  (
    for j in 0 1 2 3; do
      seed=$((9340 + w * 4 + j))
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -1
    done
  ) &
done
wait
echo ""
echo "完成：g93*.npz 共 $(ls -1 training_history/prior_labels/g93*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"