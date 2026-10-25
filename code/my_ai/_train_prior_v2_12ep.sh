#!/usr/bin/env bash
# v2 训练（修正映射 + 建塔0-15广播 + 解冻主干），12 epoch
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=4 OMP_NUM_THREADS=4
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
echo "############ prior v2 训练（12 epoch, 解冻主干）  $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_v2.py --pattern "g93*.npz" --epochs 12 --batch 8 \
  --unfreeze trunk --lr 1e-4 --out checkpoints/prior_v2_trunk.pt
echo "结束 $(date +%H:%M:%S)"