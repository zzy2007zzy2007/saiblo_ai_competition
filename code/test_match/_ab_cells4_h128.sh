#!/usr/bin/env bash
# 内部对打：**cells4 + every1 + H128**（A 侧） vs **cells4 + every1 + H256**（B 侧）。
#
# 动机：
#   · 水平扫描显示 MC_HORIZON=256 是离群点（vs rule_v4 0.891，而 128 有 0.984）
#   · T1 显示 MC_EVERY=1 与 5 无显著差异，但 every1 让「每回合」都仲裁 ⇒ dec_max 已到 7.53 s
#   · 所以想验证：把 H 从 256 砍到 128，在 cells4 配置下是否强度不掉（同时预计省 ~一半仲裁时间）
#
# 两侧完全相同，只差 AZAI_MC_HORIZON ⇒ 干净的单变量对照。64 局，8 并行，约 40 分钟。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS=$(seq 171 234 | tr '\n' ' ')

BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_ROOTONLY=1;AZAI_MC_EVERY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000"
A_H128="${BASE};AZAI_MC_HORIZON=128"
B_H256="${BASE};AZAI_MC_HORIZON=256"

echo "等待对亚军（cells4）结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 500); do
  marker=$(ls -d training_history/runs/*vs_runnerup_cells4_v2* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "对亚军（cells4）测试结束" "$marker/output.log" 2>/dev/null; then
    echo "对亚军已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

echo ""
echo "############ cells4/every1: H128 (A) vs H256 (B)  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --seeds $SEEDS --jobs 8 \
    --out-dir training_history/ab_cells4_h128_vs_h256 \
    --flat-env="$A_H128" --mcts-env="$B_H256"
echo ""
echo "H128 vs H256 结束 ($(date +%H:%M:%S))"
