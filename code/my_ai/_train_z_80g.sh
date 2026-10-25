#!/usr/bin/env bash
# 80 局重训（解冻主干 + z-score），20 epoch，每 epoch 评估留出，存最优
#   40 局时峰值在 ep3（84.8%）；数据翻倍后看峰值是否上移、拐点是否后移
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ 80 局重训（解冻主干, 20ep, eval-every 1）  $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_z.py --pattern "g93*.npz" --epochs 20 --batch 32 \
  --lr 5e-4 --unfreeze trunk --holdout 4 --eval-every 1 --save-best \
  --out checkpoints/prior_z_80g.pt
echo "结束 $(date +%H:%M:%S)"