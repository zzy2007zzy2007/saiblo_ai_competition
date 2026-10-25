#!/usr/bin/env bash
# rollout 教师采集 v3：单线程(代码级 torch.set_num_threads) + 10 并行 + 每局打耗时
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=rollout PRIOR_MAX_ROUNDS=200
export AZAI_TORCH_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "############ rollout 教师采集 v3（10 进程 × 3 局，单线程）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7 8 9; do
  (
    for j in 0 1 2; do
      seed=$((9200 + w * 3 + j))
      t0=$(date +%s)
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -1
      echo "  [w$w] seed=$seed 用时 $(( ( $(date +%s) - t0 ) / 60 )) 分钟"
    done
  ) &
done
wait
echo ""
echo "完成：g*.npz 共 $(ls -1 training_history/prior_labels/g*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"