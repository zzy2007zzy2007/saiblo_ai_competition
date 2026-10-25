#!/usr/bin/env bash
# 全量训练 v2：60 epoch（8 epoch 时 loss 仍在降 1.55→1.32，明显欠训练）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ prior z-score 全量训练 60ep  $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_z.py --pattern "g93*.npz" --epochs 60 --batch 32 \
  --lr 5e-4 --unfreeze trunk --holdout 4 --out checkpoints/prior_z_60ep.pt
echo "结束 $(date +%H:%M:%S)"