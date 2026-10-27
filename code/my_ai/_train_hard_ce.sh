#!/usr/bin/env bash
# 纭?one-hot CE 涓ょ粍锛埼?1 / 位=3锛夛紝鏃ュ織钀界洏
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export AZAI_TORCH_THREADS=8
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
for w in 1 3; do
  echo "#### hard CE lambda=$w  $(date +%H:%M:%S)"
  "$PY" -u code/my_ai/train_prior_z.py --pattern "g93*.npz" --epochs 12 --batch 32 --lr 5e-4 \
    --unfreeze trunk --ce-weight "$w" --ce-hard --eval-every 2 \
    --out "checkpoints/prior_hard_ce_w$w.pt"
done
echo "缁撴潫 $(date +%H:%M:%S)"