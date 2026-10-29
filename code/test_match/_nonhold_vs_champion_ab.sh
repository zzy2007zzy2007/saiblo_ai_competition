#!/usr/bin/env bash
# 对手无关性：**同一套"非 HOLD 注入"去打冠军**（2026-10-29，父任务"任务 2"）
#
# ## 要回答什么
# E3 批（`docs/nonhold_inject_ab_20261029.md`）证明：把 HOLD 的功劳控制住之后，
# **只注入亚军的非 HOLD 动作**仍能把对**亚军**的胜率 16.7% → 45.8%（配对 +29.2pp，p=0.039）。
# 那是"知识"还是"针对亚军风格的补偿"？把**对手换成冠军**、**oracle 仍然是亚军**，再跑一遍：
#   · A − B ≥ +12pp（CI 不跨 0）⇒ **是知识**（不依赖对手风格）
#   · A ≈ B ⇒ 只是**针对亚军的补偿**（对冠军无用）
#   · A < B ⇒ 亚军的非 HOLD 动作对冠军**有害**（风格冲突）
#
# ## 两臂（都带 `AZAI_ALWAYS_HOLD=1`）
#   B：`AZAI_ALWAYS_HOLD=1`（不注入）
#   A：`AZAI_ALWAYS_HOLD=1` + `AZAI_ORACLE_SKIP_HOLD=1` + `AZAI_ORACLE_INJECT=<亚军 exe>`（**oracle 仍是亚军**）
#
# ## 规模 / 配置
#   seeds 7..18 × 2 座位 = 每臂 24 局（共 48），两臂并行（各 5 并发，总 10）
#   对手 = 冠军 `其他版本ai/ant-war2-magica-v3/magica_v3_O0.exe`
#   `AZAI_ORACLE_TIMEOUT=15`；BASE 不变；不开安慰剂/升级/PIN_K/MC_UPG
#   `LADDER_TIMEOUT=21600`（冠军 O0 慢 + 夜里可能休眠）、`MVS_TIMEOUT_SECONDS=300`、
#   `MVS_STDERR_SUFFIX=1`、`--retry-invalid 2 --final-sweep --watchdog-stall-min 90`
#   **不起 keepawake**（夜里断电，主动接受休眠 + 重试补跑）
#
# 用法：
#   JOBS=10 bash code/run_logged.sh nonhold_vs_champion bash code/test_match/_nonhold_vs_champion_ab.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS_PER_ARM="${JOBS_PER_ARM:-5}"
RETRY="${RETRY:-0}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r}"
ROUNDS="${ROUNDS:-}"
TAG_A="${TAG_A:-NONHOLD_VS_CHAMPION_A}"
TAG_B="${TAG_B:-NONHOLD_VS_CHAMPION_B}"

export LADDER_TIMEOUT="${LADDER_TIMEOUT:-21600}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
export MVS_STDERR_SUFFIX=1
export LADDER_EXTRA_ARGS="${LADDER_EXTRA_ARGS:---retry-invalid 2 --final-sweep --watchdog-stall-min 90}"

echo "=============================================================="
echo "[nhch] $(date '+%F %T')  A=ALWAYS_HOLD+注入(跳过HOLD, oracle=亚军)  B=ALWAYS_HOLD   对手=**冠军**"
echo "[nhch] seeds=$SEEDS（$(echo $SEEDS | wc -w) 个 token/臂）  jobs/臂=$JOBS_PER_ARM  rounds=${ROUNDS:-<完整>}"
echo "[nhch] LADDER_TIMEOUT=${LADDER_TIMEOUT}s  MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[nhch] LADDER_EXTRA_ARGS=$LADDER_EXTRA_ARGS"
echo "[nhch] ⚠️ 不起保活（夜里断电；休眠靠 retry/sweep 补）"
echo "=============================================================="

run_arm() {          # $1 = A|B
    local arm="$1"
    if [ "$arm" = "A" ]; then
        TAG="$TAG_A" EXP=E3 OPP=champion SEEDS="$SEEDS" JOBS="$JOBS_PER_ARM" RETRY="$RETRY" \
            ROUNDS="$ROUNDS" \
            AZAI_ALWAYS_HOLD=1 AZAI_ORACLE_SKIP_HOLD=1 AZAI_ORACLE_REBUILD_MAX=6 \
            AZAI_ORACLE_RESTART_EVERY=0 \
            bash code/test_match/_us_vs_external.sh > "match_results/ladder_logs/_nhch_arm_${arm}.out" 2>&1
    else
        TAG="$TAG_B" EXP=E1 OPP=champion SEEDS="$SEEDS" JOBS="$JOBS_PER_ARM" RETRY="$RETRY" \
            ROUNDS="$ROUNDS" AZAI_ALWAYS_HOLD=1 AZAI_ORACLE_SKIP_HOLD=0 \
            bash code/test_match/_us_vs_external.sh > "match_results/ladder_logs/_nhch_arm_${arm}.out" 2>&1
    fi
    echo "[nhch] 臂 $arm 结束 rc=$?  $(date '+%F %T')"
}

mkdir -p match_results/ladder_logs
run_arm A &
PA=$!
run_arm B &
PB=$!
wait "$PA"; wait "$PB"

echo ""
echo "=============================================================="
echo "[nhch] 两臂结束 $(date '+%F %T')"
echo "=============================================================="

# ── 读数（注意：**必须显式 --seeds 7..18**，否则 `_batch_table.py` 只算 7..14 的 16 局）──
for T in "$TAG_A" "$TAG_B"; do
    echo ""
    echo "################  读数 $T  ################"
    "$P" -u code/test_match/_batch_table.py --tag "$T" \
        --stderr "match_results/us_vs_external/$T/stderr" --opp-label champion \
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
    --json match_results/us_vs_external/_nonhold_vs_champion_pair.json || true
echo ""
echo "################  六项指标（HOLD 率 / 升级 / 注入构成）  ################"
"$P" -u code/test_match/_ahold_report.py --tag-a "$TAG_A" --tag-b "$TAG_B" \
    --out match_results/us_vs_external/_nonhold_vs_champion_report.md \
    --json match_results/us_vs_external/_nonhold_vs_champion_report.json || true
echo "[nhch] 完成 $(date '+%F %T')"
