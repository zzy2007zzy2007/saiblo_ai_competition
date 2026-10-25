#!/usr/bin/env bash
# 对照：只训位置头（1,560 参数） + z-score 损失，40 epoch，每 2 评估
#   —— 14,308 局面 / 54 万参数会过拟合；只训头是天然的强正则
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ 只训头 + z-score（40ep, 40局）  $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_z.py --pattern "g93[0-3]*.npz" --epochs 40 --batch 32 \
  --lr 1e-3 --unfreeze head --holdout 4 --eval-every 2 --save-best \
  --out checkpoints/prior_z_head.pt
echo "结束 $(date +%H:%M:%S)"