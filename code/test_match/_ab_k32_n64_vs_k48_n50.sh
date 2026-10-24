#!/usr/bin/env bash
# 用户的论点：先验省下的预算换成**采样数**
#   A = k=32 + 先验 + 预算78000 ⇒ 有效 N=64（拉满）
#   B = k=48 + 先验 + 预算39000 ⇒ 有效 N=50
# 同为先验加权、同为随机化 rollout + 前缀树
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_CKPT=checkpoints/prior_distill_mcts16.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=17"
echo "############ A: k32+N64 vs B: k48+N50（先验加权，随机化 rollout） 32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2020 2035 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k32_n64_vs_k48_n50 \
  --flat-env="$B;AZAI_K=32;AZAI_FLAT_ROUNDS=78000" \
  --mcts-env="$B;AZAI_K=48;AZAI_FLAT_ROUNDS=39000"
echo ""
echo "采样vs候选数 对照结束 ($(date +%H:%M:%S))"