#!/usr/bin/env bash
# **广度还能不能继续换强度：k=320 vs k=160**（2026-10-23）
#
# 已有读数：
#   · k=24（采样16）→ k=160（采样1）：**65.6%**（21W-11L，血量差 +5.19，CI[+1.3,+9.0]）显著；
#   · 衰减随机 rollout（只对手/双方）≈ 50% ⇒ 随机性这条路关闭，预算全给广度。
# 成本：k=160 ≈ 930 ms/决策 ⇒ k=320 ≈ 1.8 s（仍在 10 s 上限内很多）。
#
# 本臂：A = k=320 / 采样1   vs   B = k=160 / 采样1   （唯一变量：候选束数量）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1380 1395 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUTS=1"
A_K320="$BASE;AZAI_K=320"
B_K160="$BASE;AZAI_K=160"


echo ""
echo "############ 广度加倍：k=320(A) vs k=160(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_k320_vs_k160 \
    --flat-env="$A_K320" --mcts-env="$B_K160"
echo ""
echo "k320 对照结束 ($(date +%H:%M:%S))"
