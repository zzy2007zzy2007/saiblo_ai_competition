#!/usr/bin/env bash
# **搜索到底值多少：k=160（完整排序） vs k=1（几乎不搜索）**（2026-10-23）
#
# 为什么做这个：今天内部一路提升（值网仲裁 71.9%、广度 k=160 65.6%、h16 打平 h64），
# 但**打亚军始终 6.2%**（今早 3.1% → 6.2% → 6.2%），内部收益完全没传导到绝对强度。
# 在决定"写快速模拟器提速（路线A）"还是"蒸馏/重训（路线B）"之前，必须先量清楚：
#   **我们现在的搜索，相对于"直接信策略网的 argmax"，究竟值多少？**
#
# A = k=160（当前冠军，完整排序）   B = k=1（只取 _expand 的第一个候选 ≈ 策略网 argmax）
# 32 局配对（16 图 × 2 座位），jobs=8。
#
# 判读：
#   · A 明显 > B ⇒ 搜索有用 ⇒ 瓶颈是"搜索量不足" ⇒ 路线 A（快速模拟器，把 160 提到 1000+）
#   · A ≈ B     ⇒ 搜索没用 ⇒ 瓶颈是"评估器/网络判断不了" ⇒ 路线 B（蒸馏/重训）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1420 1435 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_ROLLOUTS=1"
A_SEARCH="$BASE;AZAI_K=160"
B_NOSEARCH="$BASE;AZAI_K=1"

echo "############ 搜索价值：k=160(A) vs k=1(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_search_value_k1 \
    --flat-env="$A_SEARCH" --mcts-env="$B_NOSEARCH"
echo ""
echo "搜索价值对照结束 ($(date +%H:%M:%S))"
