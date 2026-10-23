#!/usr/bin/env bash
# **候选宽度：cells 4 → 8**（直接检验用户的机制模型："提升来自候选动作更好"）
#
# 机制模型（用户 2026-10-23 提出，与全部实测一致）：
#   估值不是靠"算得更准"直接变强，而是通过**改变第一级筛出的候选**影响实力。
#   两级结构：MC 仲裁选"类" → flat rollout 只在类内选位置 ⇒ 第一级决定候选池。
#   证据：类头塌缩(61% 闪电)、建塔位置图无信号(+0.046)、cells0→4 **+75%**、仲裁 quiet→net **+19pp**。
#
# 本臂：A = cells8（类内候选格 8 个）  vs  B = cells4（当前最优）
#   其余完全相同：值网仲裁@h64、不升级、flat rollout。
#   预测（若"候选更好"仍是瓶颈）：A > 50%；若已饱和：A ≈ 50% 或略低（多测 4 格会摊薄采样）。
#
# 32 局配对（16 图 × 2 座位），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1060 1075 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=64;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_C8="$BASE;AZAI_MC_CELLS=8"
B_C4="$BASE;AZAI_MC_CELLS=4"

echo "等待「新配方打亚军」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*VS_RU_newcfg* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "新配方打亚军结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ 候选宽度：cells8(A) vs cells4(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_cells8_vs_cells4 \
    --flat-env="$A_C8" --mcts-env="$B_C4"
echo ""
echo "cells 宽度对照结束 ($(date +%H:%M:%S))"
