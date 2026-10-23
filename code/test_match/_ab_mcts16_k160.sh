#!/usr/bin/env bash
# MCTS 教师先验在**部署口径 k=160** 下的验收（2026-10-23 夜最后一测）
# 背景：k=24 下 4 局版 62.5% 但 16 局版 50.0%（复现失败）。部署用 k=160，
#       若先验在 k=160 能省广度，才可能有部署价值。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=160;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
echo "############ MCTS先验(k160) vs 冠军(k160)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 1900 1915 | tr '\n' ' ') --jobs 8 \
    --out-dir training_history/ab_mcts16_k160 \
    --flat-env="AZAI_CKPT=checkpoints/prior_distill_mcts16.pt;$COMMON" \
    --mcts-env="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$COMMON"
echo ""
echo "k160 验收结束 ($(date +%H:%M:%S))"