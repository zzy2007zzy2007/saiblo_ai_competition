#!/usr/bin/env bash
# 关键对照：k=24 下"先验选候选" vs "随机选候选"（同网、同算力、同 k）
#   A = f10 先验加权采样（TCLASS=0.5 / TPOS=1.0 / TEMP=1e-6）
#   B = 均匀随机采样（TCLASS/TPOS/TEMP = 1e5）⇒ 先验被抹平
# 判读：A 明显 > B ⇒ "缩 k 不掉强度"是先验的功劳
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_CKPT=checkpoints/prior_distill_f10.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
echo "############ k=24: 先验选(A) vs 随机选(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2300 2315 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k24_prior_vs_random \
  --flat-env="$B;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6" \
  --mcts-env="$B;AZAI_TCLASS=100000;AZAI_TPOS=100000;AZAI_TEMP=100000"
echo ""
echo "k24 先验vs随机 结束 ($(date +%H:%M:%S))"