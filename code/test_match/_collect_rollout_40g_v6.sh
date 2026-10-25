#!/usr/bin/env bash
# rollout 教师采集 v6：双方都记 + 每回合出标签 + 200 回合
#   每局 400 样本（P0/P1 各 200）⇒ 40 局 = 16,000 局面
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=rollout PRIOR_MAX_ROUNDS=200 PRIOR_TEACHER_EVERY=1
export AZAI_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "############ rollout 教师采集 v6（10 进程 × 4 局，双方都记）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7 8 9; do
  (
    for j in 0 1 2 3; do
      seed=$((9300 + w * 4 + j))
      t0=$(date +%s)
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -1
      echo "  [w$w] seed=$seed 用时 $(( ( $(date +%s) - t0 ) / 60 )) 分钟"
    done
  ) &
done
wait
echo ""
echo "完成：g*.npz 共 $(ls -1 training_history/prior_labels/g*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"