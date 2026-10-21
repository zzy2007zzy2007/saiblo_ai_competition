#!/usr/bin/env bash
# 后续两个**内部正面对打**（避免 vs rule_v4 的天花板效应）：
#   T1: flat + MC_EVERY=1  vs  flat + MC_EVERY=5      → MC 覆盖度的真实增益
#   T2: flat + every1 + MC_CELLS=4  vs  flat + every1 → 同一类多位置（用户提案）
#
# 背景：vs rule_v4 上 A3=0.891、A4=1.000 已见底，分不出差距 ⇒ 改用自我对打。
# 每个 64 局，8 并行，seed 107..170（与之前不重叠）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 107 170 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
FLAT_E5="${BASE};AZAI_MC_EVERY=5;AZAI_MC_CELLS=0"
FLAT_E1="${BASE};AZAI_MC_EVERY=1;AZAI_MC_CELLS=0"
FLAT_E1C4="${BASE};AZAI_MC_EVERY=1;AZAI_MC_CELLS=4"

# ── 等水平扫描结束 ─────────────────────────────────────────────────────────
echo "等待水平扫描结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 300); do
  marker=$(ls -d training_history/runs/*ab_mc_horizon* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "水平扫描结束" "$marker/output.log" 2>/dev/null; then
    echo "水平扫描已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ T1: flat/Every1 (A侧) vs flat/Every5 (B侧)  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_t1_every1_vs_every5 \
    --flat-env="$FLAT_E1" --mcts-env="$FLAT_E5"

echo ""
echo "############ T2: flat/Every1/Cells4 (A侧) vs flat/Every1 (B侧)  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_t2_cells4_vs_cells0 \
    --flat-env="$FLAT_E1C4" --mcts-env="$FLAT_E1"

echo ""
echo "T1/T2 全部结束 ($(date +%H:%M:%S))"
