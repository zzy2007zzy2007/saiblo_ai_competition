#!/usr/bin/env bash
# 对手 = **亚军本体**，我们 = cells4 + every1 + H256 + **MC_UPG=1（显式升级候选）**。
#
# 动机（2026-10-21 诊断）：cells4/H256 打亚军 **0 胜 17 负**，逐局统计显示
#   · 我们出招：建塔 49.4% / 降级 45.4% / 闪电 5.1% / **升级 0.0%**
#   · 亚军出招：降级 45.7% / 建塔 42.6% / **升级 9.0%** / 闪电 2.7%
# ⇒ 我们的塔永远停在 1 级，对方升到 2–3 级。`mc_upg>0` 会在 MC 菜单里**显式构造
#   UPGRADE_TOWER 候选**（代码注释：decoder 在"格子已有塔"时给的是降级，所以升级只能这样进菜单）。
#
# 这是单变量改动：只加 AZAI_MC_UPG=1，其余与 0-17 那场逐字相同、同 32 个种子 ⇒ 可直接配对比较。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r"

echo "等待统一搜索对照结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 500); do
  marker=$(ls -d training_history/runs/*ab_unified_v3* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "统一搜索对照结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_HORIZON=256 AZAI_MC_ROOTONLY=1
export AZAI_MC_CELLS=4 AZAI_MC_UPG=1
# 2026-10-22：仲裁改用 net 模式（打一手→32 回合→值网），消除 quiet 的 0.05 量化平局
export AZAI_MC_SCORE=net AZAI_MC_NET_SAMPLES=4 AZAI_MC_HORIZON=32
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat
export AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64 AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

echo ""
echo "############ cells4 + every1 + H256 + MC_UPG=1  vs 亚军 32 局  $(date +%H:%M:%S)"
$PY -u _tmp_ladder.py --tag="VS_RUNNERUP_upg_net" --jobs=8 \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
    $SEEDS
echo ""
echo "cells4+UPG1 测试结束 ($(date +%H:%M:%S))"
