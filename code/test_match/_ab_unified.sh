#!/usr/bin/env bash
# 内部对打：**统一动作搜索**（A 侧） vs **当前最强 flat+cells4**（B 侧）。
#
# A 侧（unified）：不做类仲裁，直接枚举原子动作（建塔/每塔每个升级步/降级/闪电/HOLD）
#   每个动作 1 次 rollout（h=32）→ 剩余预算复评 top-8。
#   已验证：真实中局能枚举出 8 个升级候选并**选中升级**（+0.477 vs 建塔 +0.463）。
# B 侧（flat）：cells4 + MC_EVERY=1 + H256 的现有最强配置。
#
# 这是"两条路线谁更强"的直接对照；两侧共用同一 ckpt / 同一引擎 / 同一种子。
# ⚠️ 2026-10-21 v2：补全动作枚举（此前只枚举 4/9 种操作，漏 EMP/Deflector/Evasion 与两种基地升级）。
#    第 1 版（v1，seeds 299-362）跑出的 14 局因动作集不完整而作废。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 427 490 | tr '\n' ' ')

A_UNIFIED="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_SEARCH=unified;AZAI_UNI_HORIZON=32;AZAI_UNI_CAP=256;AZAI_UNI_TOPK=8;AZAI_UNI_ROUNDS=39000;AZAI_UNI_MINSAMP=30;AZAI_UNI_MAXLIGHT=32;AZAI_FLAT_BATCH=32"
B_FLAT="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_ROOTONLY=1;AZAI_MC_EVERY=1;AZAI_MC_CELLS=4;AZAI_MC_HORIZON=256;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"

echo "等待仲裁评估器对照结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 500); do
  marker=$(ls -d training_history/runs/*ab_arbiter_net_v3* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "仲裁评估器对照结束" "$marker/output.log" 2>/dev/null; then
    echo "仲裁对照已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ 统一搜索 (A) vs flat+cells4 (B) 64 局  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_unified_vs_cells4_v3 \
    --flat-env="$A_UNIFIED" --mcts-env="$B_FLAT"
echo ""
echo "统一搜索对照结束 ($(date +%H:%M:%S))"
