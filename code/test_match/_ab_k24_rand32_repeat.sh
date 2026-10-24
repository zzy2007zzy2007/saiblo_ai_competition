#!/usr/bin/env bash
# #8 复现"随机化 rollout 更好"（新种子，n 从 32 提到 64 合并）
#   A = k=24 + f10 先验 + 随机化（decay C=.3/k=1，只对手，前缀树，N=32）
#   B = k=24 + f10 先验 + 确定性（idle，N=1）
# 判据：与既有 ab_decay_s32（59.4%）合并后 n=64，合并 ≥57% 才算确认
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_CKPT=checkpoints/prior_distill_f10.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
RA="$B;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=32;AZAI_FLAT_SEED=23"
DE="$B;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
echo "############ #8 复现: k24 随机化N32(A) vs 确定性N1(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2400 2415 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k24_rand32_repeat \
  --flat-env="$RA" --mcts-env="$DE"
echo ""
echo "#8 结束 ($(date +%H:%M:%S))"