#!/usr/bin/env bash
# 对照：分离"k=48 本身（候选少+采样多）"与"先验好坏"
#   A = k=48 + 原网   B = k=160 + 原网
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
C="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=17"
echo "############ 对照: k48+原网 vs k160+原网（随机化 rollout） 32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 1960 1975 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k48_vs_k160_baseprior \
  --flat-env="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=48;$C" \
  --mcts-env="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=160;$C"
echo ""
echo "对照结束 ($(date +%H:%M:%S))"