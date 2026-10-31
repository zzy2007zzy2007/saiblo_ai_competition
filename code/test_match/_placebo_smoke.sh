#!/usr/bin/env bash
# 安慰剂臂的**冒烟**（2026-10-28；报告 docs/placebo_inject_ab_20261028.md）：
# 一局短对局（canonical 配置 + `AZAI_ORACLE_PLACEBO=1`），只为验四件事：
#   ① stderr 里出现 `[placebo] 安慰剂注入已启用`（不是静默不启用）；
#   ② 每决策一条 `[placebo] inject round=… action=… dup=… new=… chosen=… hit=…`；
#   ③ 收尾 `注入统计` 有 new/dup/hit/kinds/gen_ms 且 `illegal=0`；
#   ④ 裁判侧 `illegal-summary illegal_total=0`、`verdict=` 不是 illegal_strict。
# 用法： bash code/run_logged.sh placebo_smoke bash code/test_match/_placebo_smoke.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
BR=code/test_match/az_bridge_ai.py
SEED="${SEED:-9001}"
ROUNDS="${ROUNDS:-40}"
CANON="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_SEED=53"
PH="$CANON;AZAI_ORACLE_PLACEBO=1"

echo "[smoke] P 臂 env = $PH"
echo "[smoke] B 臂 env = $CANON"
echo "[smoke] seed=$SEED rounds=$ROUNDS  $(date '+%Y-%m-%d %H:%M:%S')"
"$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" \
    --ai0="$BR" --ai1="$BR" --ai0-env="$PH" --ai1-env="$CANON" "$SEED"
RC=$?
echo "[smoke] rc=$RC  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "================ 安慰剂侧 stderr 摘要（ai0）================"
f=$(ls -1t match_results/ai0_az_bridge_ai_seed${SEED}*.stderr.log 2>/dev/null | head -1)
echo "[smoke] stderr = $f"
if [ -n "$f" ]; then
    grep -c "inject round=" "$f" | sed 's/^/[smoke] 每决策注入日志行数 = /'
    grep -h "安慰剂注入已启用" "$f" | head -1
    grep -h "inject round=" "$f" | head -5
    grep -h "注入统计" "$f" | tail -1
    grep -h "illegal-summary" "$f" | tail -1
fi
echo "================ 基准臂 stderr 摘要（ai1，应无 [placebo] 行）================"
g=$(ls -1t match_results/ai1_az_bridge_ai_seed${SEED}*.stderr.log 2>/dev/null | head -1)
echo "[smoke] stderr = $g"
if [ -n "$g" ]; then
    echo "[smoke] ai1 里 [placebo] 行数 = $(grep -c '\[placebo\]' "$g" || true)（应为 0）"
fi
exit $RC
