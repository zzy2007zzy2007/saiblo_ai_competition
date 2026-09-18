#!/usr/bin/env bash
# 补上"未训练基线"在 seed 128..255 上的成绩 —— 否则新 seed 上训后的 49.2% 没法解读
# （0-127 上是 35.2%，seed 集间波动就有 14pp，必须同 seed 比）。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
for s in 128 192; do
  echo "########## baseline（未训练）seed=$s ##########"
  "$PY" -u code/my_ai/az_intent/eval.py --checkpoint "$CK" --opponent rule_v4 \
      --games 64 --workers 16 --seed "$s" --bundle-mcts --native-engine --iterations 256 \
      --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only \
      --skip-single-candidate
done
