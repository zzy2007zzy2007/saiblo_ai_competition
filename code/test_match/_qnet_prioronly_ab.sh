#!/usr/bin/env bash
# 纯类先验 vs 原版 unified（terminal Q 网，2026-11-02）。
# A 臂 = AZAI_CAND_MODE=unified + AZAI_QNET_CLASS_PRIOR=1（**只加先验**，无 s2_mix、无游标改动）。
# B 臂 = 原版 unified（AZAI_CAND_MODE=unified + BASE，无任何增强开关）。
# 目的：直接测"Q 网先验单独有没有让 unified 变强"（vs 原版 unified 的胜率 delta）。
#
# 用法：
#   SEEDS="100 101" bash code/test_match/_qnet_prioronly_ab.sh   # 4 局/臂
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-2}"
SEEDS="${SEEDS:-100 101}"
RETRY="${RETRY:-1}"
GAME_TIMEOUT_S="${GAME_TIMEOUT_S:-10800}"
OUTDIR="match_results/qnet_combo_ab/prioronly"
TAG="qprioronly"

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[qprioronly] !! 读不到 BASE" >&2; exit 2; }
case "$BASE" in
    *AZAI_CAND_MODE*) echo "[qprioronly] !! BASE 里已含 AZAI_CAND_MODE ⇒ 停" >&2; exit 2 ;;
esac

CKPT="training_history/qnet_action/qnet_terminal.pt"
mkdir -p "$OUTDIR"
echo "=============================================================="
echo "[qprioronly] 纯类先验 vs 原版 unified  seeds=[$SEEDS] jobs=$JOBS"
echo "[qprioronly] A: class_prior=1 (terminal)；B: 原版 unified（无增强）"
echo "[qprioronly] BASE 与 _us_vs_external.sh:68 逐字相同（${#BASE} 字符）"
echo "[qprioronly] $(date '+%F %T')"
echo "=============================================================="

FLAT_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1;AZAI_QNET_CKPT=$CKPT;AZAI_QNET_CLASS_PRIOR=1"
MCTS_ENV="$BASE;AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1"
export MVS_STDERR_SUFFIX=1
export AB_SEARCH_SUFFIX_TAG="_$TAG"
"$P" -u code/test_match/ab_search_modes.py \
    --seeds $SEEDS --jobs "$JOBS" --paired-sides \
    --out-dir "$OUTDIR" \
    --flat-env "$FLAT_ENV" --mcts-env "$MCTS_ENV" \
    --resume --retry-invalid "$RETRY" --game-timeout-s "$GAME_TIMEOUT_S" \
    2>&1 | tee "$OUTDIR/_run.log"
rc=${PIPESTATUS[0]}
echo "[qprioronly] 结束 rc=$rc  $(date '+%F %T')"
echo "[qprioronly] 分析："
echo "  $P -u code/test_match/analyze_qnet_prior_ab.py --out-dir $OUTDIR --stderr-tag $TAG --label-a 'A(pure-classprior)' --seeds $(echo $SEEDS | tr ' ' ',') --json $OUTDIR/summary.json"
