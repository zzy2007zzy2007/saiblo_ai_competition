#!/usr/bin/env bash
# **全枚举候选 vs 先验采样 k=160**（2026-10-23）
#
# 诊断依据（code/test_match/diag_prior_quality.py，枚举 300 候选全部跑 rollout 估值）：
#   · 先验 vs 估值 Spearman = **-0.015 / -0.015 / -0.037**（基本无相关，甚至反相关）
#   · 估值最优动作的**先验排名 = 293/300、152/300、217/300**
#   · recall@64 = **全部未命中**（最优动作几乎从不落在先验前 64 名）
#   ⇒ 候选筛选（动作类头 + 位置网建塔通道）基本是坏的；k=160 是"对坏先验做暴力兜底"，
#     而且按先验加权采样，最好的动作可能根本没进候选集。
#
# 本臂：A = 全枚举（k=300，采样1，h16）  B = k=160（现状）
#   成本：k=300 约 2.8 s/决策（k=160 是 1.53 s），10 s 预算内。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1580 1595 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
A_ENUM="$BASE;AZAI_K=300"
B_K160="$BASE;AZAI_K=160"

echo "############ 全枚举 k=300(A) vs k=160(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_enum300_vs_160 \
    --flat-env="$A_ENUM" --mcts-env="$B_K160"
echo ""
echo "全枚举对照结束 ($(date +%H:%M:%S))"
