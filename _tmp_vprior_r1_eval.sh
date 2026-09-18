#!/usr/bin/env bash
# 第一轮价值偏好位置网的**行为判据**：把它当先验（默认 --pos-prior policy，
# 但 ckpt 的 pos_state 已换成训好的位置网），k=24 打 rule_v4。
# 对照（同 seed、可逐局配对，均在 value_prior_rule_v4 / value_prior_verify 里）：
#   未训练位置网  21.1%（128 局）
#   价值 oracle  37.5%（128 局）
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/vprior/posnet_r1.pt
for s in 0 64; do
  echo "########## value-prior posnet r1, seed=$s ##########"
  "$PY" -u code/my_ai/az_intent/eval.py \
      --checkpoint "$CK" --opponent rule_v4 --games 64 --workers 16 --seed "$s" \
      --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 \
      --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only --skip-single-candidate
done
