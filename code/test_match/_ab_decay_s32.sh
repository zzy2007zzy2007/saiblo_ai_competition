#!/usr/bin/env bash
# **随机 rollout + 衰减 + 大采样（32 次/候选）**（2026-10-23 用户建议）
#
# 已有读数（随机 rollout 系列）：
#   · 恒定 p=0.5 双方   → 34.4%（CI[-12.3,-2.6]）显著差
#   · 双方 + 衰减       → 31.2%（CI[-11.3,-2.4]）显著差
#   · **只对手 + 衰减，采样 8** → **50.0%（CI[-5.3,+5.4]）中性**
#
# 用户的判断：采样 8 太少 —— 随机 rollout 方差大，8 次均值仍可能让 argmax 被噪声带偏；
# "中性"可能是**信号被噪声淹没**，而不一定没信号。成本允许更大的采样：
#   只对手+衰减 采样 8 = 997 ms，采样 16 = 1808 ms，采样 32 ≈ 3.5 s（10 s 上限内）。
#
# 本臂：A = k=160 + 只对手 + decay(C=.6,k=3) + **采样 32**
#       B = 当前冠军 = k=160 + 静止 + 采样 1
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1460 1475 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_K=160;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_DECAY32="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=32;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.6;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ACT_PROB=0.6;AZAI_FLAT_SEED=17"
B_CHAMP="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"

echo "等待「搜索价值 k=1」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*ab_search_value_k1* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "搜索价值对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ 随机衰减 采样32(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_decay_s32 \
    --flat-env="$A_DECAY32" --mcts-env="$B_CHAMP"
echo ""
echo "采样32对照结束 ($(date +%H:%M:%S))"
