#!/usr/bin/env bash
# τ 的**证伪检验**：分析（见 docs/az_value_prior_plan.md §0.2 与 train_value_prior.py 文件头）
# 预测 τ 近似空操作 —— 因为采样侧解码器会重新 z-score，τ 与地图的绝对尺度一起被约掉。
# 用同一份数据、同一个 --seed 0（与 posnet_r1 的划分/训练顺序逐字相同），**唯一变量 = τ**。
# 对照：posnet_r1（τ=1.0）seed0-63 = 32.8%、128 局 = 35.2%。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
FIX=training_history/az_fixed
VP=training_history/vprior

for t in 0.25 0.5; do
  echo "########## 训练 target_tau=$t ##########"
  "$PY" -u code/my_ai/az_intent/train_value_prior.py --ckpt "$FIX/three_mix_r10p_vw_pol_frozen.pt" \
      --data "$VP/vp_400.npz" --epochs 60 --batch-size 256 --lr 1e-3 \
      --target-tau "$t" --t-pos 1.0 --seed 0 --out "$VP/posnet_tau$t.pt"
  for s in 0 64; do
    echo "########## 评测 target_tau=$t seed=$s ##########"
    "$PY" -u code/my_ai/az_intent/eval.py --checkpoint "$VP/posnet_tau$t.pt" --opponent rule_v4 \
        --games 64 --workers 16 --seed "$s" --bundle-mcts --native-engine --iterations 256 \
        --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 1.0 --search-mode pos-only \
        --skip-single-candidate
  done
done
