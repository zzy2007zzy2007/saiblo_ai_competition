#!/usr/bin/env bash
# #10-a 复测: k=48 + f10 先验 + 随机化 N=50（原读数 40.6%，判据：合并 n=64 后是否仍 <45%）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
A="AZAI_CKPT=checkpoints/prior_distill_f10.pt;$B;AZAI_K=48;AZAI_FLAT_ROUNDS=39000$rnd"
CH="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$B;AZAI_K=160;AZAI_FLAT_ROUNDS=39000"
echo "############ #10-a 复测: k48+f10先验(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2480 2495 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k48_prior_repeat \
  --flat-env="$A" --mcts-env="$CH"
echo ""
echo "#10-a 结束 ($(date +%H:%M:%S))"