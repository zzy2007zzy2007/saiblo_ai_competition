#!/usr/bin/env bash
# 价值偏好位置网**第二轮（迭代/自提升）**：用第一轮的 posnet_r1 自己采数据再训一遍。
#
# 三个问题一次回答：
#   ① 能不能复利（项目的老问题：之前 5 轮位置网循环都没复利）
#   ② 更好的局面分布（r1 的裸策略比旧基线强得多：raw 镜像 1.4688）是否给出更好的目标
#   ③ 不做数据缩放，先只换"谁在采"这一个变量
#
# 对照（同 seed、可逐局配对）：
#   旧基线（未训练位置网）  256 局 18.4%（47W/256）
#   第一轮 posnet_r1        256 局 42.2%（108W/256）
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
VP=training_history/vprior

echo "########## 1) 用 posnet_r1 采 400 局（约 1 小时）##########"
"$PY" -u code/my_ai/az_intent/collect_value_prior.py \
    --checkpoint "$VP/posnet_r1.pt" --games 400 --workers 16 --seed 920401 \
    --out "$VP/vp_r2.npz"

echo "########## 2) 训练 posnet_r2（warm start 自 r1 的 pos 网）##########"
"$PY" -u code/my_ai/az_intent/train_value_prior.py --ckpt "$VP/posnet_r1.pt" \
    --data "$VP/vp_r2.npz" --epochs 60 --batch-size 256 --lr 1e-3 \
    --target-tau 1.0 --t-pos 1.0 --seed 0 --out "$VP/posnet_r2.pt"

echo "########## 3) 评测 posnet_r2（4 批 seed，共 256 局）##########"
for s in 0 64 128 192; do
  echo "---------- seed=$s ----------"
  "$PY" -u code/my_ai/az_intent/eval.py --checkpoint "$VP/posnet_r2.pt" --opponent rule_v4 \
      --games 64 --workers 16 --seed "$s" --bundle-mcts --native-engine --iterations 256 \
      --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only \
      --skip-single-candidate
done

echo "########## 4) raw 镜像：r2 裸策略 vs r1 裸策略 ##########"
"$PY" -u _tmp_position_choice_ab.py --mode raw --checkpoint "$VP/posnet_r2.pt" \
    --opponent-checkpoint "$VP/posnet_r1.pt" --pairs 32 --workers 8 --seed 0
