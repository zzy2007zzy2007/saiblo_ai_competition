#!/usr/bin/env bash
# 干净对照：只在"是否会话引擎 RNG"上不同（2026-10-28）
#   B（基准，固定）: random + decay(1.0,3) + RNG_PERTURB=0 + FLAT_SEED=53
#   A（被测）      : random + decay(1.0,3) + RNG_PERTURB=1 + FLAT_SEED=53
# 其余完全相同；16 局（地图 3400-3407 × 2 座位），--jobs 8
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
ARM_B="$BASE;AZAI_FLAT_RNG_PERTURB=0"
ARM_A="$BASE;AZAI_FLAT_RNG_PERTURB=1"

echo "############ 干净对照：RNG 扰动 开 vs 关（其余全同）  $(date +%H:%M:%S)"
"$PY" -u code/test_match/ab_search_modes.py --paired-sides \
  --seeds $(seq 3400 3407 | tr '\n' ' ') --jobs 8 \
  --out-dir "match_results/rng_perturb_clean" \
  --flat-env="$ARM_A" --mcts-env="$ARM_B" 2>&1 | tail -4
echo "结束 $(date +%H:%M:%S)（读法：flat 胜率 = 被测臂 A（扰动开）的得分）"
