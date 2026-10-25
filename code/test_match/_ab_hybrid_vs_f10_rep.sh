#!/usr/bin/env bash
# 复现 hybrid vs 纯 f10（上一批 19W-13L=59.4%，判据：合并 n=64 后是否仍 >=55%）
#   同配置、同 FLAT_SEED=53，只换种子（2800-2815）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_K=48;AZAI_FLAT_ROUNDS=49920;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=53"
A="AZAI_CKPT=checkpoints/prior_z_hybrid_light.pt;$COMMON"
B="AZAI_CKPT=checkpoints/prior_distill_f10.pt;$COMMON"
echo "############ 复现 hybrid vs 纯f10（32 局配对, seeds 2800-2815）  $(date +%H:%M:%S)"
"$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2830 2845 | tr '\n' ' ') --jobs 10 \
  --out-dir training_history/ab_hybrid_vs_f10_rep2 \
  --flat-env="$A" --mcts-env="$B"
echo "结束 $(date +%H:%M:%S)"
