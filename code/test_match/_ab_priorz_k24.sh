#!/usr/bin/env bash
# 新先验实战验收：k=24 + prior_z_80g.best（留出@24=85.0%） vs 冠军(k=160 确定性)
#   与 ab_distill_f10_k24 同协议，f10 当时是 16W-14L = 53.3%（+0.3hp）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle"
A="AZAI_CKPT=checkpoints/prior_z_80g.pt.best.pt;$COMMON;AZAI_K=24;AZAI_FLAT_ROLLOUTS=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6"
B="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$COMMON;AZAI_K=160;AZAI_FLAT_ROLLOUTS=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6"
echo "############ 新先验 k=24 A/B（32 局配对）  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2660 2675 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_priorz_k24 \
  --flat-env="$A" --mcts-env="$B"
echo ""
echo "结束 $(date +%H:%M:%S)"