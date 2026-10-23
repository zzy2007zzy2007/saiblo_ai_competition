#!/usr/bin/env bash
# 一键：训练教师版位置先验 → k=24 配对验收（2026-10-23）
# 用法: bash code/test_match/_train_and_eval_prior.sh <pattern> <tag> <seed_start>
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
PATTERN=${1:-m*.npz}
TAG=${2:-mcts}
S0=${3:-1820}
CKPT="checkpoints/prior_distill_${TAG}.pt"

echo "=== [1/2] 训练（pattern=$PATTERN, 8 epoch） $(date +%H:%M:%S)"
$PY -u code/my_ai/train_prior_distill.py --pattern "$PATTERN" --frac 1.0 --epochs 8 --out "$CKPT" 2>&1 | tail -4

echo ""
echo "=== [2/2] k=24 配对验收（$CKPT vs 冠军） $(date +%H:%M:%S)"
COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
SEEDS=$(seq $S0 $((S0+15)) | tr '\n' ' ')
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir "training_history/ab_${TAG}_k24" \
    --flat-env="AZAI_CKPT=$CKPT;$COMMON" \
    --mcts-env="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$COMMON"
echo ""
echo "完成 ($(date +%H:%M:%S))"