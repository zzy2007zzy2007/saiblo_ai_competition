#!/usr/bin/env bash
# 定位最优 epoch：60 epoch，每 3 个评估留出集，按留出 @24 存最优
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ 最优 epoch 定位（eval-every 3, save-best）  $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_z.py --pattern "g93*.npz" --epochs 60 --batch 32 \
  --lr 5e-4 --unfreeze trunk --holdout 4 --eval-every 3 --save-best \
  --out checkpoints/prior_z_best.pt
echo "结束 $(date +%H:%M:%S)"