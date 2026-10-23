#!/usr/bin/env bash
# **保守随机 + 大采样 + 前缀树**（2026-10-23 用户方向）
#
# 依据链：
#   1. 采样 8（C=.6/k=3，只对手） = 50.0%（中性）
#   2. 采样 32（同配置）           = **59.4%**（19W-13L，血量差+3.69，CI[-1.0,+8.4]）
#      ⇒ **随机性有信号，之前是采样不足把信号淹了**（+9.4pp）
#   3. 前缀树记忆化已实现并验证（分数与朴素版逐位一致）：
#      高共享配置 decay(C=.3,k=1)（共享 85%）下 N=32 省 **38%**；低共享（43%）反而慢。
#   ⇒ 本臂把三个改进叠起来：**保守随机（共享高、轨迹贴近"静止"）+ N=64 + 前缀树**。
#
# A = 只对手 + decay(C=0.3,k=1) + 采样 64 + 前缀树
# B = 当前冠军（静止 + 采样 1）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1500 1515 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_K=160;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_SEED=17"
B="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"

echo "############ 保守随机+N64+前缀树(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_random_bign_tree \
    --flat-env="$A" --mcts-env="$B"
echo ""
echo "保守随机大采样对照结束 ($(date +%H:%M:%S))"
