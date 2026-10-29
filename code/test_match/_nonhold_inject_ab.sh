#!/usr/bin/env bash
# 非 HOLD 外部动作定价：`AZAI_ORACLE_SKIP_HOLD` A/B（2026-10-29，父任务批准）
#
# ## 要回答什么
# 上一轮（`docs/always_hold_ab_20261029.md`）：把"恒为候选的 HOLD"单独补上只拿回 **+12.5pp**
# （≈ 亚军注入总增益 +37.5pp 的 1/3），剩下 2/3 没解释。这一步把 **HOLD 的功劳控制住**
# （两臂都开 `AZAI_ALWAYS_HOLD=1`），只让**非 HOLD 的外部动作**进候选：
#   · A 臂（被测）：`AZAI_ALWAYS_HOLD=1` + `AZAI_ORACLE_INJECT=<亚军>` + `AZAI_ORACLE_SKIP_HOLD=1`
#     （oracle 给 `()` 时**不注入**，只计 `skipped_hold`）
#   · B 臂（基准）：`AZAI_ALWAYS_HOLD=1`（不注入）
# ⇒ 两臂之差 = "真外部动作"的净价值（不需要冠军、不需要新 oracle）。
#
# ## 预注册判读（**跑之前登记，别事后改**）
#   · **A − B ≥ +12pp 且 CI 不跨 0** ⇒ 非 HOLD 外部动作**确实有独立价值**（"知识注入"叙事部分成立）
#   · **A ≈ B（|Δhp| < 5 且 CI 跨 0）** ⇒ +37.5pp **基本全是"能空过"的功劳**，外部动作知识价值很小
#   · **A < B** ⇒ 外部非 HOLD 动作有害（要考虑是不是亚军风格与我们不兼容）
#
# ## 规模与配置
#   seeds **7..18**（12 个地图）× 2 座位 = **每臂 24 局**（共 48 局），两臂**并行**（各 5 并发）
#   其余全用 BASE（`_inject_vs_runnerup.sh:45` 那串，脚本内有逐字断言）；`FLAT_SEED=53`
#   不开：安慰剂 / 升级候选 / PIN_K / MC_UPG / oracle 主动重启（亚军是无状态查询器，不需要）
#
# ## 必须报的指标（写进 `docs/nonhold_inject_ab_20261029.md`）
#   1. 配对胜负/血量差（A−B，逐 token）+ 胜率 Wilson CI + 符号检验 p
#   2. 按座位拆分（上一轮增益全在 @P1）
#   3. 注入内容构成：`asks / skipped_hold / new / dup / hit_new` + **非 HOLD 注入的实际采纳率**
#   4. 两臂的 HOLD 率、降级 churn
#   5. `verdict` 全 engine、`illegal=0`、零影响证明
#
# 用法：
#   JOBS=10 bash code/run_logged.sh nonhold_inject_ab bash code/test_match/_nonhold_inject_ab.sh
#   冒烟：ROUNDS=12 SEEDS="7 7r" JOBS_PER_ARM=1 bash code/test_match/_nonhold_inject_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
# 两臂并行 ⇒ 每臂并发 = 总并发/2（默认 10 ⇒ 每臂 5）
JOBS_PER_ARM="${JOBS_PER_ARM:-5}"
RETRY="${RETRY:-0}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r}"
ROUNDS="${ROUNDS:-}"
TAG_A="${TAG_A:-NONHOLD_INJECT_A}"
TAG_B="${TAG_B:-NONHOLD_INJECT_B}"

# 过夜/长跑三件套（任务书口径）
export LADDER_TIMEOUT="${LADDER_TIMEOUT:-10800}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
export MVS_STDERR_SUFFIX=1
export LADDER_EXTRA_ARGS="${LADDER_EXTRA_ARGS:---retry-invalid 2 --final-sweep --watchdog-stall-min 90}"

echo "=============================================================="
echo "[nh-ab] $(date '+%F %T')  A=ALWAYS_HOLD+注入(跳过HOLD)  B=ALWAYS_HOLD"
echo "[nh-ab] seeds=$SEEDS（$(echo $SEEDS | wc -w) 个 token/臂）  jobs/臂=$JOBS_PER_ARM  rounds=${ROUNDS:-<完整>}"
echo "[nh-ab] LADDER_TIMEOUT=${LADDER_TIMEOUT}s  MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[nh-ab] LADDER_EXTRA_ARGS=$LADDER_EXTRA_ARGS"
echo "[nh-ab] ⚠️ 不起保活（AC 供电；休眠靠 retry/sweep 补）"
echo "=============================================================="

run_arm() {          # $1 = A|B
    local arm="$1"
    if [ "$arm" = "A" ]; then
        # EXP=E3 = 注入亚军 vs 亚军（`_us_vs_external.sh` 里那一条）；SKIP_HOLD 由环境传进去
        TAG="$TAG_A" EXP=E3 OPP=runnerup SEEDS="$SEEDS" JOBS="$JOBS_PER_ARM" RETRY="$RETRY" \
            ROUNDS="$ROUNDS" \
            AZAI_ALWAYS_HOLD=1 AZAI_ORACLE_SKIP_HOLD=1 AZAI_ORACLE_REBUILD_MAX=6 \
            AZAI_ORACLE_RESTART_EVERY=0 \
            bash code/test_match/_us_vs_external.sh > "match_results/ladder_logs/_nh_arm_${arm}.out" 2>&1
    else
        TAG="$TAG_B" EXP=E1 OPP=runnerup SEEDS="$SEEDS" JOBS="$JOBS_PER_ARM" RETRY="$RETRY" \
            ROUNDS="$ROUNDS" AZAI_ALWAYS_HOLD=1 AZAI_ORACLE_SKIP_HOLD=0 \
            bash code/test_match/_us_vs_external.sh > "match_results/ladder_logs/_nh_arm_${arm}.out" 2>&1
    fi
    echo "[nh-ab] 臂 $arm 结束 rc=$?  $(date '+%F %T')"
}

mkdir -p match_results/ladder_logs
run_arm A &
PA=$!
run_arm B &
PB=$!
wait "$PA"; wait "$PB"

echo ""
echo "=============================================================="
echo "[nh-ab] 两臂结束 $(date '+%F %T')"
echo "=============================================================="
for f in match_results/ladder_logs/_nh_arm_A.out match_results/ladder_logs/_nh_arm_B.out; do
    echo ""
    echo "################  $f 的尾巴  ################"
    tail -12 "$f" 2>/dev/null || true
done

# ── 读数 ────────────────────────────────────────────────────────────────────
for T in "$TAG_A" "$TAG_B"; do
    echo ""
    echo "################  读数 $T  ################"
    # 2026-10-29：**必须显式给 --seeds**：`_batch_table.py` 默认只认 7..14（16 局），
    #   本轮是 7..18（24 局）⇒ 不给就会把后 8 局静默漏掉（第一版就是这样，n 显示 16）。
    "$P" -u code/test_match/_batch_table.py --tag "$T" \
        --stderr "match_results/us_vs_external/$T/stderr" --opp-label runnerup \
        --seeds 7 8 9 10 11 12 13 14 15 16 17 18 \
        --markdown-out "match_results/us_vs_external/$T/table.md" || true
    "$P" -u code/test_match/analyze_inject_vs_runnerup.py \
        --arm "$T=match_results/ladder_logs/$T" \
        --stderr "$T=match_results/us_vs_external/$T/stderr" \
        --json "match_results/us_vs_external/$T/summary.json" || true
done
echo ""
echo "################  配对（A−B，同 token）  ################"
"$P" -u code/test_match/analyze_inject_vs_runnerup.py \
    --arm "A=match_results/ladder_logs/$TAG_A" \
    --stderr "A=match_results/us_vs_external/$TAG_A/stderr" \
    --arm "B=match_results/ladder_logs/$TAG_B" \
    --stderr "B=match_results/us_vs_external/$TAG_B/stderr" \
    --pair A B \
    --json match_results/us_vs_external/_nonhold_pair.json || true
echo ""
echo "################  六项指标（HOLD 率 / churn / 注入构成）  ################"
"$P" -u code/test_match/_ahold_report.py --tag-a "$TAG_A" --tag-b "$TAG_B" \
    --out match_results/us_vs_external/_nonhold_report.md \
    --json match_results/us_vs_external/_nonhold_report.json || true
echo "[nh-ab] 完成 $(date '+%F %T')"
