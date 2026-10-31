#!/usr/bin/env bash
# 「默认行为不变」的**对局级**证据（2026-10-28；报告 docs/placebo_inject_ab_20261028.md §4）
#
# 要证明的东西：**`AZAI_ORACLE_PLACEBO` 不设时，加这个开关对默认行为零影响**。
# 做法（任务书要求的口径 = "同一 seed 的 RESULT 逐字段相同"）：
#   1. 改动前 = `git show <改动前 commit>:code/test_match/az_bridge_ai.py` 的**原样副本**
#      （`_az_bridge_ai_head.py`，同目录 ⇒ import 路径与原来完全一致）；
#   2. 改动后 = 工作区的 `code/test_match/az_bridge_ai.py`；
#   3. 两局**同一 seed、同一配置**（部署 canonical：ITERS=256/K=24/FLAT_ROUNDS=39000、
#      FLAT_SEED=53、`--rounds=200`），两局**并发**跑（2 局 × 2 个 AI = 4 个进程，机器 16 核）；
#   4. 用 `compare_result_lines.py` 逐字段比 `RESULT` 行 + AI 侧 `非法操作自检汇总`
#      （= 发出去的 op 序列/holds 全同）。⚠️ 两个文件名不同 ⇒ `p0=/p1=/winner=` 里的 AI 标签
#      会被归一化成 `<AI>`；`shadow_ms=` 这类**墙钟**读数也会剔掉（那不是行为）。
#
# 用法：
#   bash code/run_logged.sh placebo_zero_impact bash code/test_match/_placebo_zero_impact.sh
#   bash code/test_match/_placebo_zero_impact.sh --compare-only   # 复用已跑出来的两份日志，只做比较
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
SEED="${SEED:-7}"
ROUNDS="${ROUNDS:-200}"
# 改动前的 commit：`f3e46d9` = 安慰剂代码**之前**的那次提交（当时 `az_bridge_ai.py` 还没有 placebo）。
# 之所以钉死而不是用 `HEAD`：这次改动一旦提交，`HEAD` 里就有 placebo 了，`git show HEAD:` 会自欺欺人。
HEAD_REF="${HEAD_REF:-f3e46d9}"
OUT="match_results/placebo_zero_impact"
BEFORE="code/test_match/_az_bridge_ai_head.py"
AFTER="code/test_match/az_bridge_ai.py"
mkdir -p "$OUT"

if [ "${1:-}" != "--compare-only" ]; then
    # ① 取"改动前"的代码：`HEAD_REF` 的原样副本
    git show "$HEAD_REF:$AFTER" > "$BEFORE" || { echo "[zi] !! 取 $HEAD_REF 版本失败"; exit 2; }
    echo "[zi] 改动前副本 = $BEFORE （git $HEAD_REF）；改动后 = $AFTER"
    echo "[zi] before 字节=$(wc -c < "$BEFORE")  after 字节=$(wc -c < "$AFTER")"
    git diff --stat "$HEAD_REF" -- "$AFTER" | sed 's/^/[zi] diff: /'

    # 部署 canonical 配置（与 `_illegal_smoke.sh` 的 CANON 逐字相同 + flat 搜索）
    CANON="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_K=24;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=5;AZAI_MC_HORIZON=256;AZAI_MC_ROOTONLY=1;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_ROLLOUTS=16;AZAI_FLAT_HORIZON=64;AZAI_FLAT_BATCH=32;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_SEED=53"
    # 两个 AI 都是桥接（同 config）⇒ 局面演化只由 seed + 配置决定，两边应当逐字节相同
    # ⚠️ 绝不设 AZAI_ORACLE_PLACEBO / AZAI_ORACLE_INJECT（这就是"默认"的定义）
    unset AZAI_ORACLE_PLACEBO AZAI_ORACLE_PLACEBO_HOLD_P AZAI_ORACLE_INJECT 2>/dev/null || true

    echo "[zi] seed=$SEED rounds=$ROUNDS  $(date '+%Y-%m-%d %H:%M:%S')  两局并发"
    "$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" \
        --ai0="$BEFORE" --ai1="$BEFORE" --ai0-env="$CANON" --ai1-env="$CANON" "$SEED" \
        > "$OUT/before_seed${SEED}.log" 2>&1 &
    PID_B=$!
    "$P" -u code/test_match/match_via_sdk.py --rounds="$ROUNDS" \
        --ai0="$AFTER" --ai1="$AFTER" --ai0-env="$CANON" --ai1-env="$CANON" "$SEED" \
        > "$OUT/after_seed${SEED}.log" 2>&1 &
    PID_A=$!
    wait $PID_B; RC_B=$?
    wait $PID_A; RC_A=$?
    echo "[zi] rc: before=$RC_B after=$RC_A  $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo ""
echo "================ 逐字段比较（compare_result_lines.py）================"
"$P" -u code/test_match/compare_result_lines.py \
    "$OUT/before_seed${SEED}.log" "$OUT/after_seed${SEED}.log"
RC_DIFF=$?
echo "[zi] 比较退出码=$RC_DIFF（0 = 默认行为逐字段不变）"
exit $RC_DIFF
