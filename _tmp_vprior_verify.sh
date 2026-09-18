#!/usr/bin/env bash
# 训练层面的独立复核（第一轮价值偏好位置网 35.2% vs 基线 21.1%）。
# 三件独立的事：
#   A. raw 镜像：训后的**裸策略** vs 训前的裸策略（完全不搜索）—— 增益是否只在搜索侧
#   B. 复训（--seed 1：不同 train/val 划分与训练顺序）+ 同 128 局评测
#   C. 换一批没见过的 game seed（128..255）评同一个 ckpt —— 测泛化与评测噪声
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
FIX=training_history/az_fixed
VP=training_history/vprior

echo "########## A. raw 镜像：训后裸策略 vs 训前裸策略 ##########"
"$PY" -u _tmp_position_choice_ab.py --mode raw \
    --checkpoint "$VP/posnet_r1.pt" \
    --opponent-checkpoint "$FIX/three_mix_r10p_vw_pol_frozen.pt" \
    --pairs 32 --workers 8 --seed 0

echo
echo "########## B. 复训 --seed 1 ##########"
"$PY" -u code/my_ai/az_intent/train_value_prior.py --ckpt "$FIX/three_mix_r10p_vw_pol_frozen.pt" \
    --data "$VP/vp_400.npz" --epochs 60 --batch-size 256 --lr 1e-3 --tau 1.0 --t-pos 1.0 \
    --seed 1 --out "$VP/posnet_r1s1.pt"

for s in 0 64; do
  echo "########## B 评测：posnet_r1s1 seed=$s ##########"
  "$PY" -u code/my_ai/az_intent/eval.py --checkpoint "$VP/posnet_r1s1.pt" --opponent rule_v4 \
      --games 64 --workers 16 --seed "$s" --bundle-mcts --native-engine --iterations 256 \
      --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only \
      --skip-single-candidate
done

for s in 128 192; do
  echo "########## C. posnet_r1 在新 seed=$s 上（64 局）##########"
  "$PY" -u code/my_ai/az_intent/eval.py --checkpoint "$VP/posnet_r1.pt" --opponent rule_v4 \
      --games 64 --workers 16 --seed "$s" --bundle-mcts --native-engine --iterations 256 \
      --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only \
      --skip-single-candidate
done
