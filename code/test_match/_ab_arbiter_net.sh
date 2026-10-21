#!/usr/bin/env bash
# **对照实验（用户 2026-10-21 提出，优先级高于统一搜索）**：
#   我们的 rollout（打一手 → h 回合 → 叶值网）能不能胜任**类仲裁**？
#
#   A 侧 = cells4 + every1 + H32 + **MC_SCORE=net**（我们的 rollout 当仲裁器, 8 次采样）
#   B 侧 = cells4 + every1 + H32 + MC_SCORE='' （现有安静 rollout 仲裁, 1 次采样）
#   两侧只差仲裁评估器 ⇒ 干净单变量。
#
# 为什么先做这个：统一搜索把"类仲裁"和"位置评估"合成同一套评估器，
# 若直接测它而效果不好，无法区分是「评估器不行」还是「合并结构不行」。
# 这个对照把评估器单独拎出来验证。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 235 298 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_ROOTONLY=1;AZAI_MC_EVERY=1;AZAI_MC_CELLS=4;AZAI_MC_HORIZON=32;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
A_NET="${BASE};AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=8"
B_QUIET="${BASE}"

echo "等待 VS_RUNNERUP_cells4_upg1 结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 500); do
  marker=$(ls -d training_history/runs/*vs_runnerup_upg1* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "cells4+UPG1 测试结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ 仲裁评估器对照: MC_SCORE=net (A) vs quiet (B)  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_arbiter_net_vs_quiet \
    --flat-env="$A_NET" --mcts-env="$B_QUIET"
echo ""
echo "仲裁评估器对照结束 ($(date +%H:%M:%S))"
