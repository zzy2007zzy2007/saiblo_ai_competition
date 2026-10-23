#!/usr/bin/env bash
# **省下来的预算花在哪：采样 ×4**（h16 下）
#
# 前置结论（2026-10-23）：
#   · 深度 h64 → h16：**46.9%（15W-17L，CI[-5.6,+3.7]）= 打平** ⇒ 64 回合的深度冗余；
#   · 宽度 cells4 → cells8：56.2%（CI 跨 0）= 基本饱和 ⇒ 加宽这条轴吃完了。
#   ⇒ 省下的 4× 预算优先买**采样**（每个候选测更多次 ⇒ 均值更准）。
#
# 本臂：A = h16 + 采样 64   vs   B = h16 + 采样 16
#   · 每组预算：A 16×64 = 1024 回合，B 16×16 = 256 回合；
#   · 对照意义：h64×16（旧）也是 1024 回合 ⇒ A 与旧配置**同预算**，
#     若 A ≈ 旧配置（h64×16 对 h16×16 是 53.1% 的方向未测）… 这里直接用 B 作参照，
#     看"多 4 倍采样"能否带来可测的提升。
#
# 32 局配对（16 图 × 2 座位），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1140 1155 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_S64="$BASE;AZAI_FLAT_ROLLOUTS=64"
B_S16="$BASE;AZAI_FLAT_ROLLOUTS=16"

echo "############ 预算花法：采样×4(A) vs 现状(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_samples64_vs_16 \
    --flat-env="$A_S64" --mcts-env="$B_S16"
echo ""
echo "采样数对照结束 ($(date +%H:%M:%S))"
