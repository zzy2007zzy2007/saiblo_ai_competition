#!/usr/bin/env bash
# 加宽/加深/正则 三组对照（串行，GPU）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ A: width=128 blocks=6 wd=1e-4  $(date +%H:%M:%S)"
"$PY" -u code/my_ai/train_prior_wide.py --width 128 --blocks 6 --epochs 20 --wd 1e-4 --tag A_w128
echo "############ B: width=128 blocks=8 wd=1e-4  $(date +%H:%M:%S)"
"$PY" -u code/my_ai/train_prior_wide.py --width 128 --blocks 8 --epochs 20 --wd 1e-4 --tag B_w128_b8
echo "############ C: width=64 blocks=6 wd=1e-4（正则对照）  $(date +%H:%M:%S)"
"$PY" -u code/my_ai/train_prior_wide.py --width 64 --blocks 6 --epochs 20 --wd 1e-4 --tag C_w64_wd
echo "全部结束 $(date +%H:%M:%S)"
