#!/usr/bin/env bash
# **对局级零影响**：`AZAI_CAND_MODE` 不设（= pin）时，同 seed 同历史下
# `RESULT` + 双方 `--trace-ops` op 序列**逐字节相同**（2026-10-29）。
#
# 做法（与 `_only_extra_zero_impact.sh` §② 同一个模式）：
#   before = `code/test_match/_az_bridge_ai_preunifiedcand.py`
#            （**加代码那一刻**的 `az_bridge_ai.py` 原样副本；只把 build_engine 里
#             `from ...flat_rollout_greedy import FlatRolloutGreedy` 换成
#             "加载加代码前那份 `_flat_rollout_greedy_preunifiedcand.py`"）
#   after  = `code/test_match/az_bridge_ai.py`（现行；**不设 AZAI_CAND_MODE**）
#   两边各自 self-play（ai0=ai1=同一份代码）⇒ 代码行为相同则两局逐字节相同。
#
# 用法：
#   ROUNDS=120 bash code/test_match/_cand_mode_zero_impact_game.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
SEED="${SEED:-8}"
ROUNDS="${ROUNDS:-120}"
OUT="match_results/unified_cand"
BEFORE="code/test_match/_az_bridge_ai_preunifiedcand.py"
AFTER="code/test_match/az_bridge_ai.py"
mkdir -p "$OUT"

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
[ -n "$BASE" ] || { echo "[zig] !! 读不到 BASE" >&2; exit 2; }
[ -f "$BEFORE" ] || { echo "[zig] !! 缺改动前副本 $BEFORE" >&2; exit 2; }

echo "=============================================================="
echo "[zig] 对局级零影响：self-play before/before vs after/after"
echo "[zig] seed=$SEED rounds=$ROUNDS  BASE=$BASE" | head -c 400; echo
echo "[zig] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

export MVS_STDERR_SUFFIX=1
PIDS=()
for SIDE in before after; do
    AI="$BEFORE"; [ "$SIDE" = "after" ] && AI="$AFTER"
    export DSH_PROGRESS_SUFFIX="_ZI_${SIDE}"
    export DSH_PROGRESS_TAG="CANDMODE_ZI"
    if [ "${SKIP_GAMES:-0}" = "1" ]; then
        echo "[zig] SKIP_GAMES=1 ⇒ 跳过 $SIDE 臂（复用已有日志）"
        continue
    fi
    "$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" --trace-ops \
        --ai0="$AI" --ai1="$AI" --ai0-env="$BASE" --ai1-env="$BASE" "$SEED" \
        > "$OUT/zero_impact_${SIDE}_seed${SEED}.log" 2>&1 &
    PIDS+=("$!:$SIDE")
done
echo "[zig] 已起 ${#PIDS[@]} 局：${PIDS[*]}"
RC_ALL=0
for e in "${PIDS[@]}"; do
    pid="${e%%:*}"; side="${e#*:}"
    wait "$pid"; rc=$?
    echo "[zig]   $side pid=$pid rc=$rc  $(date '+%H:%M:%S')"
    [ "$rc" -ne 0 ] && RC_ALL=1
done

echo ""
echo "=============================================================="
echo "[zig] ① RESULT 逐字段（compare_result_lines.py）"
echo "=============================================================="
# ⚠️ `compare_result_lines.py` 是工作区里另一批实验留下的工具（未跟踪）。缺它就跳过这一步，
#    **不能**把它算成"零影响失败"（② 的逐字节比较已经覆盖 RESULT 行）。
if [ -f code/test_match/compare_result_lines.py ]; then
    "$P" -u code/test_match/compare_result_lines.py \
        "$OUT/zero_impact_before_seed${SEED}.log" "$OUT/zero_impact_after_seed${SEED}.log"
    RC1=$?
    echo "[zig] compare rc=$RC1（0 = 逐字段相同）"
    [ "$RC1" -ne 0 ] && RC_ALL=1
else
    echo "[zig] ⚠️ 缺 code/test_match/compare_result_lines.py ⇒ 跳过①（由下面的②逐字节覆盖）"
fi

echo ""
echo "=============================================================="
echo "[zig] ② 双方 op 序列逐字节（--trace-ops 行）"
echo "=============================================================="
for f in "$OUT/zero_impact_before_seed${SEED}.log" "$OUT/zero_impact_after_seed${SEED}.log"; do
    # `--trace-ops` 的每回合原文行：`  [round N] p0(<label>) 原文='...' -> [Operation(...)]`
    # 把 AI 标签归一化成 `<AI>`（before 臂的文件名不同，标签必然不同，但那不是行为差异）
    grep -aE "^\s+\[round [0-9]+\] p[01]\(" "$f" | sed -E 's/p([01])\([^)]*\)/p\1(<AI>)/' > "${f%.log}.ops" || true
done
B="$OUT/zero_impact_before_seed${SEED}.ops"
A="$OUT/zero_impact_after_seed${SEED}.ops"
echo "[zig] before ops 行数=$(wc -l < "$B")  after ops 行数=$(wc -l < "$A")"
if cmp -s "$B" "$A"; then
    echo "[zig] ✅ op 序列**逐字节相同**（sha256 $(sha256sum "$B" | cut -d' ' -f1 | cut -c1-16)）"
else
    echo "[zig] ❌ op 序列有差异："
    diff "$B" "$A" | head -20
    RC_ALL=1
fi
if cmp -s <(grep -a "^  RESULT" "$OUT/zero_impact_before_seed${SEED}.log" \
             | sed -E 's/p0=[^ ]*/p0=<AI>/;s/p1=[^ ]*/p1=<AI>/;s/winner=[^ ]*/winner=<AI>/') \
          <(grep -a "^  RESULT" "$OUT/zero_impact_after_seed${SEED}.log" \
             | sed -E 's/p0=[^ ]*/p0=<AI>/;s/p1=[^ ]*/p1=<AI>/;s/winner=[^ ]*/winner=<AI>/'); then
    echo "[zig] ✅ RESULT 行（AI 标签归一化后）**逐字节相同**"
else
    echo "[zig] ❌ RESULT 行不同"
    RC_ALL=1
fi
echo "[zig] 总退出码=$RC_ALL（0 = 对局级零影响成立）"
exit "$RC_ALL"
