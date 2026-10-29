#!/usr/bin/env bash
# 过夜批次：E4 / E4-rand × {对手=冠军, 对手=亚军} = 4 个 tag × 16 局 = 64 局（2026-10-29）
#
# 顺序（**先跑最有价值的那一对**：E4 vs E4-rand 的配对检验只需要同一对手的两臂）：
#   1. E4   vs 冠军    （ONLYEXTRA_VS_CHAMPION）
#   2. E4R  vs 冠军    （ONLYEXTRA_RAND_VS_CHAMPION）
#   3. E4   vs 亚军    （ONLYEXTRA_VS_RUNNERUP）
#   4. E4R  vs 亚军    （ONLYEXTRA_RAND_VS_RUNNERUP）
# 每跑完一个 tag 就立刻出读数（逐局表 + 覆盖度/分差诊断），**不用等 64 局全完**。
#
# 为什么按这个顺序：夜里可能断电/休眠。先跑冠军那两臂 ⇒ 即使后面没跑完，
# 也能拿到"同一对手、同 seeds/座位"的 E4 vs E4-rand 配对读数（那才是预注册的主判据）。
#
# 用法：
#   JOBS=10 bash code/run_logged.sh e4_overnight_all bash code/test_match/_e4_overnight_all.sh
#
# 环境（都可覆盖）：JOBS（默认 10）、SEEDS、RETRY（默认 0 —— 重试交给 ladder 自己的
#   `--retry-invalid 2 --final-sweep`，那一层是进程内重跑，比外层 while 循环更省）。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-10}"
RETRY="${RETRY:-0}"
export JOBS RETRY
export SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"

TAGS="ONLYEXTRA_VS_CHAMPION ONLYEXTRA_RAND_VS_CHAMPION ONLYEXTRA_VS_RUNNERUP ONLYEXTRA_RAND_VS_RUNNERUP"

run_one() {          # $1 = 脚本  $2 = OPP
    local script="$1" opp="$2"
    echo ""
    echo "################  $(date '+%F %T')  $script  OPP=$opp  jobs=$JOBS  ################"
    OPP="$opp" bash "code/test_match/$script" || echo "[e4-all] ⚠️ $script OPP=$opp rc=$?"
}

echo "=============================================================="
echo "[e4-all] 64 局过夜批次开始 $(date '+%F %T')"
echo "[e4-all] LADDER_TIMEOUT=${LADDER_TIMEOUT:-<脚本默认 21600>}s jobs=$JOBS retry=$RETRY"
echo "[e4-all] LADDER_EXTRA_ARGS=${LADDER_EXTRA_ARGS:-<脚本默认 --retry-invalid 2 --final-sweep --watchdog-stall-min 90>}"
echo "[e4-all] oracle: REBUILD_MAX=${AZAI_ORACLE_REBUILD_MAX:-6} " \
     "RESTART_EVERY=${AZAI_ORACLE_RESTART_EVERY:-150} RESTART_EVERY_B=${AZAI_ORACLE_RESTART_EVERY_B:-0}"
echo "=============================================================="

# 前置：确认没有别人的对局进程在跑（**只看不杀**；有就停，别和人家抢 CPU）
_others="$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -match '_tmp_ladder|az_bridge_ai|ai_cpp_lure_v4|magica_v3_O0' } | Measure-Object).Count" 2>/dev/null | tr -d '\r')"
echo "[e4-all] 现存的 ladder/bridge/oracle 进程数 = ${_others:-?}（0 才好；不是 0 也**不杀**，只记一笔）"

run_one _e4_only_extra.sh       champion
run_one _e4r_only_extra_rand.sh champion
run_one _e4_only_extra.sh       runnerup
run_one _e4r_only_extra_rand.sh runnerup

echo ""
echo "=============================================================="
echo "[e4-all] 全部 4 个 tag 结束 $(date '+%F %T')"
echo "=============================================================="

# ── 读数（每个 tag：逐局表 + 覆盖度/分差诊断）──────────────────────────────
for T in $TAGS; do
    case "$T" in
        *CHAMPION*) LBL=champion ;;
        *)          LBL=runnerup ;;
    esac
    echo ""
    echo "################  读数 $T（对手=$LBL）  ################"
    "$P" -u code/test_match/_batch_table.py --tag "$T" \
        --stderr "match_results/us_vs_external/$T/stderr" --opp-label "$LBL" \
        --markdown-out "match_results/us_vs_external/$T/table.md" || true
    "$P" -u code/test_match/analyze_xoracle.py --tag "$T" \
        --stderr "match_results/us_vs_external/$T/stderr" \
        --out "match_results/xoracle/$T" || true
done

echo ""
echo "[e4-all] 配对分析（E4 vs E4-rand，同一对手、同 seeds/座位）："
for OPP in CHAMPION RUNNERUP; do
    "$P" -u code/test_match/analyze_inject_vs_runnerup.py \
        --arm "E4=match_results/ladder_logs/ONLYEXTRA_VS_$OPP" \
        --stderr "E4=match_results/us_vs_external/ONLYEXTRA_VS_$OPP/stderr" \
        --arm "E4R=match_results/ladder_logs/ONLYEXTRA_RAND_VS_$OPP" \
        --stderr "E4R=match_results/us_vs_external/ONLYEXTRA_RAND_VS_$OPP/stderr" \
        --pair E4 E4R \
        --json "match_results/us_vs_external/_pair_${OPP}.json" || true
done
echo "[e4-all] 完成 $(date '+%F %T')"
