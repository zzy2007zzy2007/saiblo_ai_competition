#!/usr/bin/env bash
# 可学性判定：200 个局面 × 2000 步（lr 1e-3），看 loss 能否显著低于 1.0
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=6
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ 过拟合检验 200局面×2000步 lr1e-3  $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_z.py --overfit-states 200 --steps 2000 --batch 8 --lr 1e-3 --epochs 99
echo "结束 $(date +%H:%M:%S)"