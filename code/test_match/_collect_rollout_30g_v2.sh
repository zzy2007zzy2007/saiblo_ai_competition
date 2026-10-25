#!/usr/bin/env bash
# rollout 教师采集 v2：修掉线程超订（每进程原先吃 6.5-6.9 核，4 进程共 27 核 > 16 核）
#   · OMP/MKL/OPENBLAS 线程锁 1 ⇒ 每进程约 1 核
#   · 10 并行（10 核左右，留余量）
#   · 每局结束打印耗时，便于估算总时长
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PRIOR_TEACHER=rollout PRIOR_MAX_ROUNDS=200
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "############ rollout 教师采集 v2（10 进程 × 3 局，单线程）  $(date +%H:%M:%S)"
for w in 0 1 2 3 4 5 6 7 8 9; do
  (
    for j in 0 1 2; do
      seed=$((9200 + w * 3 + j))
      t0=$(date +%s)
      $PY -u code/test_match/collect_prior_labels.py "$seed" 2>&1 | tail -1
      echo "  [w$w] seed=$seed 用时 $(( (`$(date +%s)` - t0) / 60 )) 分钟"
    done
  ) &
done
wait
echo ""
echo "完成：g*.npz 共 $(ls -1 training_history/prior_labels/g*.npz 2>/dev/null | wc -l) 个  $(date +%H:%M:%S)"