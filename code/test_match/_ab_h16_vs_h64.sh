#!/usr/bin/env bash
# **第一步：flat rollout 的深度能不能砍短？**（单变量 FLAT_HORIZON）
#
# 动机（2026-10-23 成本分析）：
#   我们的单个 rollout = 64 回合 × 62.7 µs ≈ **4.0 ms**；
#   亚军 = 8 回合 × 6.5 µs ≈ **51.7 µs** ⇒ 它每决策能测 14,350 个 rollout，我们只能测 384。
#   而我们有**值网**可以在叶节点代替一部分搜索深度 ⇒ 64 回合很可能是浪费。
#
# 单变量：A 的 FLAT_HORIZON=16（省 4×），B=64（现状）。采样数、候选数、MC 配置完全相同。
#   · A ≈ B ⇒ 深度确实多余 ⇒ 砍短成立，省下的预算可换采样或换候选（第二步）；
#   · A 明显 < B ⇒ 深度有用，退到 32 再试。
#
# 32 局配对（16 图 × 2 座位），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 1100 1115 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_MC_HORIZON=64;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A_H16="$BASE;AZAI_FLAT_HORIZON=16"
B_H64="$BASE;AZAI_FLAT_HORIZON=64"

echo "等待「cells 宽度对照」结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 200); do
  marker=$(ls -d training_history/runs/*ab_cells8_vs_cells4* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "cells 宽度对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"; break
  fi
  sleep 30
done

echo ""
echo "############ 深度：h16(A) vs h64(B)，采样/候选相同  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_h16_vs_h64 \
    --flat-env="$A_H16" --mcts-env="$B_H64"
echo ""
echo "rollout 深度对照结束 ($(date +%H:%M:%S))"
