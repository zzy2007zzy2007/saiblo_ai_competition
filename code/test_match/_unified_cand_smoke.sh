#!/usr/bin/env bash
# 1 局冒烟：`AZAI_CAND_MODE=unified` vs 亚军（cpp_lure_v4），BASE 配置逐字复用
# （从 `_inject_vs_runnerup.sh` 里读，不重抄 ⇒ 不会漂）。
#
# 用法：
#   ROUNDS=12 TAG=UNIFIED_CAND_SMOKE bash code/test_match/_unified_cand_smoke.sh
#   ROUNDS=512 TAG=UNIFIED_CAND_FULL bash code/test_match/_unified_cand_smoke.sh   # 跑满
#   MODE=pin ...                                                                   # 对照臂
#
# ⚠️ 只开 **1 局**（`--jobs` 概念在这里不存在：一个 match_via_sdk 进程 = 一局）⇒ 低并发。
# ⚠️ stderr 文件名加 `_UC` 后缀（`MVS_STDERR_SUFFIX=1`）⇒ **不会覆盖**别人正在写的
#    `ai0_az_bridge_ai_seed<N>.stderr.log`。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
SEED="${SEED:-8}"
ROUNDS="${ROUNDS:-512}"
TAG="${TAG:-UNIFIED_CAND_SMOKE}"
MODE="${MODE:-unified}"          # unified | pin
VERBOSE="${VERBOSE:-1}"
# 额外的臂开关（`;` 分隔，如 `AZAI_CAND_DECODE=split`）；**默认空** ⇒ 与历史冒烟逐字相同
EXTRA="${EXTRA:-}"
STDERR_SUF="${STDERR_SUF:-_UC}"   # stderr 文件名后缀（不同臂用不同后缀，互不覆盖）
OUTDIR="match_results/unified_cand"
mkdir -p "$OUTDIR"

RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
RU="$(pwd -W | sed 's#/$##')/$RU_REL"
[ -f "$RU" ] || { echo "[uc] !! 亚军 exe 不存在: $RU" >&2; exit 2; }

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
[ -n "$BASE" ] || { echo "[uc] !! 读不到 BASE" >&2; exit 2; }
ENV0="$BASE;AZAI_CAND_MODE=$MODE;AZAI_CAND_VERBOSE=$VERBOSE${EXTRA}"

export MVS_STDERR_SUFFIX=1
export DSH_PROGRESS_SUFFIX="$STDERR_SUF"
export DSH_PROGRESS_TAG="$TAG"

LOG="$OUTDIR/${TAG}_seed${SEED}.log"
echo "[uc] tag=$TAG seed=$SEED rounds=$ROUNDS mode=$MODE verbose=$VERBOSE extra=${EXTRA:-<无>}"
echo "[uc] ai0=code/test_match/az_bridge_ai.py  ai1=$RU"
echo "[uc] stdout -> $LOG ；AI stderr -> match_results/ai0_*_seed${SEED}${STDERR_SUF}.stderr.log"
echo "[uc] $(date '+%Y-%m-%d %H:%M:%S')"
"$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" --trace-ops \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" \
    --ai0-env="$ENV0" "$SEED" > "$LOG" 2>&1
RC=$?
echo "[uc] 退出码=$RC  $(date '+%Y-%m-%d %H:%M:%S')"
grep -E "^  RESULT|illegal-summary|IMPL|⚠️" "$LOG" | head -40
exit "$RC"
