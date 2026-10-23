#!/usr/bin/env bash
# **保守随机 + N=128 + 前缀树**（2026-10-23，继续加大采样）
#
# 曲线（全部 32 局配对，对照冠军）：
#   · 衰减 C=.6/k=3，只对手，N=8  ：**50.0%**（CI[-5.3,+5.4]，中性）
#   · 衰减 C=.6/k=3，只对手，N=32 ：**59.4%**（CI[-1.0,+8.4]）
#   · 衰减 C=.3/k=1，只对手，N=64 + 前缀树：**68.8%**（CI[+0.2,+10.0]）
#   ⇒ 采样越多越好；而前缀树把 N 翻倍的成本几乎吃掉
#      （N=32 朴素 48.3 分钟/局 vs N=64 带树 46.9 分钟/局）。
#
# 本臂：N=128（再翻倍），仍用 C=.3/k=1（共享率 85%，树的收益最大）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1540 1555 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_K=160;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_ROLLOUTS=128;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_SEED=17"
B="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"

echo "############ 保守随机+N128+前缀树(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_random_n128_tree \
    --flat-env="$A" --mcts-env="$B"
echo ""
echo "N128 对照结束 ($(date +%H:%M:%S))"
