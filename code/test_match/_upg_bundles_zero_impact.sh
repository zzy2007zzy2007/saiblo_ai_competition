#!/usr/bin/env bash
# 「`AZAI_UPG_BUNDLES` 不设时默认行为逐位不变」的**对局级**证据（2026-10-28）
# 报告：`docs/upgrade_bundles_20261028.md` §6。
#
# 口径（任务书要求）：**同一 seed、同一配置，开关关 vs 改动前** ⇒ `RESULT` 行逐字段相同。
#   1. 改动前 = `code/test_match/_az_bridge_ai_preupg.py`（**加代码那一刻**的工作区原样副本，
#      md5 见 §6；⚠️ 不用 `git show HEAD:` —— 那里面已经有本次改动，等于自欺欺人）；
#   2. 改动后 = `code/test_match/az_bridge_ai.py`，**不设** `AZAI_UPG_BUNDLES`（= 默认 0）；
#   3. 配置 = `_inject_vs_runnerup.sh` 里的 BASE **逐字**（正式批的同一串，含 K=48/FLAT_ROUNDS=49152）；
#   4. `--rounds=200` 截断（够跑出 200 个决策，比 §5 的完整 500 回合便宜一个量级）；
#   5. 两个 AI 都是本桥接、同 config ⇒ 局面演化只由 seed + 配置决定 ⇒ 两边应当逐字节相同；
#   6. 用 `compare_result_lines.py` 逐字段比 `RESULT` + AI 侧 `illegal-summary` 自检行。
#
# 用法：
#   bash code/run_logged.sh upg_bundles_zero_impact bash code/test_match/_upg_bundles_zero_impact.sh
#   bash code/test_match/_upg_bundles_zero_impact.sh --compare-only   # 复用已跑出来的日志，只比较
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
SEEDS="${SEEDS:-7 7r}"
ROUNDS="${ROUNDS:-200}"
OUT="match_results/upgrade_bundles/zero_impact"
# 「改动前」的两份副本（同一文件，md5 相同）：优先 `code/test_match/` 下那份（与原来的 import 路径
# 完全一致，最保险）；不在就用 `match_results/` 里的本地工件副本（`match_results/` 被 gitignore）。
BEFORE="code/test_match/_az_bridge_ai_preupg.py"
if [ ! -f "$BEFORE" ] && [ -f "match_results/upgrade_bundles/_az_bridge_ai_preupg.py" ]; then
    BEFORE="match_results/upgrade_bundles/_az_bridge_ai_preupg.py"
    echo "[zi] ⚠️ 用 match_results 里的 before 副本（注意：它离 repo 根更近，import 路径与正式版不同，仍可跑）"
fi
AFTER="code/test_match/az_bridge_ai.py"
BEFORE_MD5="8841F843C058D68564077ABC1978EC8E"   # 加代码**之前**那一份工作区文件的 md5
mkdir -p "$OUT"

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
[ -n "$BASE" ] || { echo "[zi] !! 读不到 BASE"; exit 2; }

# 「默认」的定义：绝不设这两个开关
unset AZAI_UPG_BUNDLES 2>/dev/null || true
echo "[zi] 环境自检：AZAI_UPG_BUNDLES=${AZAI_UPG_BUNDLES:-<未设=默认0>}"

RC_ALL=0
if [ "${1:-}" != "--compare-only" ]; then
    [ -f "$BEFORE" ] || { echo "[zi] !! 缺改动前副本 $BEFORE"; exit 2; }
    MD5_NOW="$(md5sum "$BEFORE" | cut -d' ' -f1 | tr 'a-f' 'A-F')"
    echo "[zi] 改动前副本 md5=$MD5_NOW （期望 $BEFORE_MD5）"
    [ "$MD5_NOW" = "$BEFORE_MD5" ] || { echo "[zi] !! 改动前副本被改过 ⇒ 比较无效，终止"; exit 3; }
    echo "[zi] before 字节=$(wc -c < "$BEFORE")  after 字节=$(wc -c < "$AFTER")"
    echo "[zi] seeds=$SEEDS rounds=$ROUNDS  $(date '+%Y-%m-%d %H:%M:%S')  每 seed 两局并发"

    PIDS=()
    for SEED in $SEEDS; do
        for SIDE in before after; do
            AI="$BEFORE"; [ "$SIDE" = "after" ] && AI="$AFTER"
            "$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" --trace-ops \
                --ai0="$AI" --ai1="$AI" --ai0-env="$BASE" --ai1-env="$BASE" "$SEED" \
                > "$OUT/${SIDE}_seed${SEED}.log" 2>&1 &
            PIDS+=("$!:$SIDE:$SEED")
        done
    done
    echo "[zi] 已起 ${#PIDS[@]} 个对局进程：${PIDS[*]}"
    for e in "${PIDS[@]}"; do
        pid="${e%%:*}"; rest="${e#*:}"
        wait "$pid"; rc=$?
        echo "[zi]   pid=$pid ($rest) rc=$rc  $(date '+%H:%M:%S')"
        [ "$rc" -ne 0 ] && RC_ALL=1
    done
    echo "[zi] 全部结束 rc=$RC_ALL  $(date '+%Y-%m-%d %H:%M:%S')"
fi

for SEED in $SEEDS; do
    echo ""
    echo "================ seed=$SEED 逐字段比较（compare_result_lines.py）================"
    "$P" -u code/test_match/compare_result_lines.py \
        "$OUT/before_seed${SEED}.log" "$OUT/after_seed${SEED}.log"
    RC=$?
    echo "[zi] seed=$SEED 比较退出码=$RC（0 = 逐字段不变）"
    [ "$RC" -ne 0 ] && RC_ALL=1
done
echo ""
echo "[zi] 总退出码=$RC_ALL（0 = 所有 seed 的默认行为逐字段不变）"
exit $RC_ALL
