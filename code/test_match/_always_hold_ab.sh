#!/usr/bin/env bash
# HOLD 恒为候选（`AZAI_ALWAYS_HOLD`）的 A/B 消融：**替代 E4**（2026-10-29，用户已批准）
#
# ## 要回答什么
# 冠军当 oracle 实测恒给 `()`（`force_pass_` 锁死，99.7-99.93%），亚军注入臂 8117 次注入里
# **84.5% 也是 HOLD**、`hit_new` 里 70.5% 是 HOLD ⇒ "注入有效"（12.5%→50%）**很可能只是**
# "把 HOLD 补进候选集"这一个效应。本脚本把它**单独**做出来（不需要任何 oracle）：
#   臂 A：`AZAI_ALWAYS_HOLD=1`（候选集里保证有一个空 bundle `()`）
#   臂 B：不设该开关（现役）
#   对手：亚军本体（`cpp_lure_v4`），seeds 7..14 × 2 座位 = 每臂 16 局（同 seeds ⇒ 可逐 token 配对）
#
# ## 预注册判据（跑之前登记，别事后改）
#   · A 复现**大部分**增益（≥35%：即 A 的胜率 ≥ 12.5% + 0.35×(50.0−12.5) ≈ 25.6%）
#     ⇒ 结论改写为：**缺陷是"类仲裁独占动/不动决策"，不是知识注入**；
#   · A 明显低于 35% ⇒ HOLD 只是部分原因，注入还有别的机制（再拆）；
#   · A < B（负）⇒ 强评估器其实不该看到 HOLD（与 70.5% 那个数矛盾，要解释）。
#
# ## 六项必须报的指标
#   1. 配对胜负/血量差（A−B，逐 token）+ 胜率 Wilson CI；并排：B 单独=12.5%/−20.12、亚军注入臂=50.0%/+0.56
#   2. HOLD 率：A 的"最终发出空 op"回合占比 vs B 的 78%（`[ahold]` 行 + 逐局 trace 两种口径）
#   3. 降级 churn：`DOWNGRADE` 占 op 比例（基线 46.0%）是否下降
#   4. MC 仲裁 vs 强评估器在"动/不动"上的一致率（`[ahold]` 的 added/chose_after_add/present_*）
#   5. `verdict` 全 engine、`illegal=0`
#   6. 零影响证明：开关关时同 seed 的 `RESULT` 逐字段相同（见 `docs/always_hold_ab_20261029.md` §0）
#
# 用法：
#   JOBS=9 bash code/run_logged.sh always_hold_ab bash code/test_match/_always_hold_ab.sh
#   冒烟：ROUNDS=12 JOBS=1 SEEDS=7 bash code/test_match/_always_hold_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-9}"
RETRY="${RETRY:-0}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"
ROUNDS="${ROUNDS:-}"

# 过夜/长跑三件套 + 超时（任务书口径）
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-10800}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
export MVS_STDERR_SUFFIX=1
export LADDER_EXTRA_ARGS="${LADDER_EXTRA_ARGS:---retry-invalid 2 --final-sweep --watchdog-stall-min 90}"

echo "=============================================================="
echo "[hold-ab] $(date '+%F %T')  arms=A(AZAI_ALWAYS_HOLD=1) / B(off)  OPP=runnerup"
echo "[hold-ab] seeds=$SEEDS  jobs=$JOBS  retry=$RETRY  rounds=${ROUNDS:-<完整>}"
echo "[hold-ab] LADDER_TIMEOUT=${LADDER_TIMEOUT}s MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[hold-ab] LADDER_EXTRA_ARGS=$LADDER_EXTRA_ARGS"
echo "[hold-ab] ⚠️ 本批次**不开**保活（AC 供电；休眠就靠 retry/sweep 补）"
echo "=============================================================="

run_arm() {          # $1 = A|B
    local arm="$1" tag
    if [ "$arm" = "A" ]; then
        export AZAI_ALWAYS_HOLD=1
        tag="${TAG_A:-ALWAYSHOLD_A}"
    else
        unset AZAI_ALWAYS_HOLD
        tag="${TAG_B:-ALWAYSHOLD_B}"
    fi
    echo ""
    echo "################  臂 $arm  TAG=$tag  $(date '+%F %T')  ################"
    TAG="$tag" EXP=E1 OPP=runnerup SEEDS="$SEEDS" JOBS="$JOBS" RETRY="$RETRY" \
        ROUNDS="$ROUNDS" bash code/test_match/_us_vs_external.sh
    echo "################  臂 $arm 结束 rc=$?  $(date '+%F %T')  ################"
}

run_arm A
run_arm B

echo ""
echo "=============================================================="
echo "[hold-ab] 两臂结束 $(date '+%F %T')"
echo "=============================================================="

# ── 读数 ────────────────────────────────────────────────────────────────────
for T in "${TAG_A:-ALWAYSHOLD_A}" "${TAG_B:-ALWAYSHOLD_B}"; do
    echo ""
    echo "################  读数 $T  ################"
    "$P" -u code/test_match/_batch_table.py --tag "$T" \
        --stderr "match_results/us_vs_external/$T/stderr" --opp-label runnerup \
        --markdown-out "match_results/us_vs_external/$T/table.md" || true
    "$P" -u code/test_match/analyze_inject_vs_runnerup.py \
        --arm "$T=match_results/ladder_logs/$T" \
        --stderr "$T=match_results/us_vs_external/$T/stderr" \
        --json "match_results/us_vs_external/$T/summary.json" || true
done

echo ""
echo "################  配对（A−B，同 token）  ################"
"$P" -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm "A=match_results/ladder_logs/${TAG_A:-ALWAYSHOLD_A}" \
    --stderr "A=match_results/us_vs_external/${TAG_A:-ALWAYSHOLD_A}/stderr" \
    --arm "B=match_results/ladder_logs/${TAG_B:-ALWAYSHOLD_B}" \
    --stderr "B=match_results/us_vs_external/${TAG_B:-ALWAYSHOLD_B}/stderr" \
    --pair A B \
    --json match_results/us_vs_external/_always_hold_pair.json || true

echo "[hold-ab] 完成 $(date '+%F %T')"
