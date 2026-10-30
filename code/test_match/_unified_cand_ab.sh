#!/usr/bin/env bash
# 统一候选生成 A/B（2026-10-31）：**A 臂 = AZAI_CAND_MODE=unified** vs **B 臂 = pin**（现役）。
#
# 为什么写独立脚本而不是复用 `_us_vs_external.sh`：
#   `_us_vs_external.sh` 有一句"先把继承的 AZAI_* 全部清掉"（第 303 行）——`AZAI_CAND_MODE`
#   会被它 unset 掉（历史上 `AZAI_ALWAYS_HOLD`/`AZAI_ORACLE_SKIP_HOLD` 就在这里踩过，必须"先抓住
#   再回填"）。与其改那个共享脚本（会影响别的批次），本脚本**自己直接调 `_tmp_ladder.py`**，
#   并且**逐臂显式 export** 环境 ⇒ 不依赖任何继承、也不会把开关漏给对手臂。
#
# 两臂之间**只差** `AZAI_CAND_MODE`（A 臂加 `=unified`；B 臂根本不设）。其余逐字相同：
#   BASE 从 `_inject_vs_runnerup.sh` 读（并断言与 `_us_vs_external.sh:68` 逐字一致）、
#   对手 = 亚军 cpp_lure_v4、seeds 7..18 × 2 座位、`--jobs 10`、`--retry-invalid 2
#   --final-sweep --watchdog-stall-min 90`、`LADDER_TIMEOUT=10800`、`MVS_TIMEOUT_SECONDS=300`。
#
# 用法：
#   bash code/test_match/_unified_cand_ab.sh              # 正式批次（A 24 局 + B 24 局）
#   PREFLIGHT_ONLY=1 bash code/test_match/_unified_cand_ab.sh   # 只跑前置校验（~1 分钟）
#   ARMS="A" bash code/test_match/_unified_cand_ab.sh     # 只跑某一臂
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-10}"
RETRY="${RETRY:-2}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r}"
ARMS="${ARMS:-A B}"
TAG_A="${TAG_A:-UNIFIED_AB_A}"
TAG_B="${TAG_B:-UNIFIED_AB_B}"
OUTROOT="match_results/unified_cand"
mkdir -p "$OUTROOT"

RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
RU="$(pwd -W | sed 's#/$##')/$RU_REL"

export LADDER_TIMEOUT="${LADDER_TIMEOUT:-10800}"
export MVS_TIMEOUT_SECONDS="${MVS_TIMEOUT_SECONDS:-300}"
export LADDER_EXTRA_ARGS="--retry-invalid $RETRY --final-sweep --watchdog-stall-min 90"

# BASE：与 `_inject_vs_runnerup.sh` / `_us_vs_external.sh` 逐字相同（两条断言，任一失败立刻停）
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_inject_vs_runnerup.sh)"
BASE_UVX="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[ab] !! 读不到 _inject_vs_runnerup.sh 的 BASE" >&2; exit 2; }
[ "$BASE" = "$BASE_UVX" ] || { echo "[ab] !! _us_vs_external.sh 的 BASE 与历史 BASE 不同 ⇒ 停" >&2; exit 3; }
# 反向：本批**必须不带**这些开关（unified 自己含 HOLD；注入/安慰剂会让两臂不可比）
BAD_SWITCHES="AZAI_ALWAYS_HOLD AZAI_ORACLE_INJECT AZAI_ORACLE_INJECT_B AZAI_ORACLE_PLACEBO AZAI_ONLY_EXTRA AZAI_ONLY_EXTRA_RANDOM AZAI_PIN_K AZAI_MC_UPG AZAI_UPG_BUNDLES"
for v in $BAD_SWITCHES; do
    if [ -n "${!v:-}" ] && [ "${!v}" != "0" ]; then
        echo "[ab] !! 环境里带着 $v=${!v} ⇒ 本批不许带（两臂会不可比），停" >&2; exit 4
    fi
done
case "$BASE" in *AZAI_CAND_MODE*) echo "[ab] !! BASE 里已经含 AZAI_CAND_MODE ⇒ 停" >&2; exit 4 ;; esac

echo "=============================================================="
echo "[ab] 统一候选 A/B   对手=亚军"
echo "[ab] A 臂（unified）= $TAG_A ；B 臂（pin）= $TAG_B"
echo "[ab] seeds=$SEEDS"
echo "[ab] jobs=$JOBS retry=$RETRY  LADDER_TIMEOUT=${LADDER_TIMEOUT}s  MVS_TIMEOUT_SECONDS=${MVS_TIMEOUT_SECONDS}s"
echo "[ab] BASE 与历史逐字相同（${#BASE} 字符，已核对 _inject_vs_runnerup.sh 与 _us_vs_external.sh 一致）"
echo "[ab] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
[ -f "$RU" ] || { echo "[ab] !! 亚军 exe 不存在: $RU" >&2; exit 2; }
for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll; do
    [ -f "$(dirname "$RU")/$d" ] || { echo "[ab] !! 缺 $d（亚军会 0xC0000139 直接死）" >&2; exit 2; }
done
echo "[ab] ✓ 亚军 exe + 3 个 DLL 都在"

# ── 前置校验（**必须先过**，否则 A 臂会静默退化成 pin，48 局白跑）──────────────
#   做两件事：① 单局 6 回合的 A 臂小跑，断言 stderr 里出现 `[cand] mode=unified`；
#            ② 断言同一局的 B 臂 stderr 里**没有** `mode=unified`（否则 B 不是 pin）。
preflight() {
    local ok=0
    for arm in A B; do
        local env0 extra tag suf
        if [ "$arm" = "A" ]; then
            extra=";AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1"; suf="_ABPRE_A"
        else
            extra=";AZAI_CAND_VERBOSE=1"; suf="_ABPRE_B"
        fi
        env0="$BASE$extra"
        # 清掉继承的 AZAI_*，再 export 本臂（与 `_us_vs_external.sh` 同样的纪律）
        local v
        for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
        local kv
        for kv in $(echo "$env0" | tr ';' '\n'); do [ -n "$kv" ] && export "$kv"; done
        unset kv
        MVS_STDERR_SUFFIX=1 DSH_PROGRESS_SUFFIX="$suf" DSH_PROGRESS_TAG="ABPRE" \
            "$P" -u code/test_match/match_via_sdk.py --rounds=6 --trace-ops \
            --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" "$SEED_PRE" \
            > "$OUTROOT/_preflight_${arm}.log" 2>&1
        local f="match_results/ai1_az_bridge_ai_seed${SEED_PRE}${suf}.stderr.log"
        [ -f "$f" ] || f="match_results/ai0_az_bridge_ai_seed${SEED_PRE}${suf}.stderr.log"
        local n_uni n_pin
        n_uni=$(grep -c "\[cand\] mode=unified" "$f" 2>/dev/null || true)
        n_pin=$(grep -c "\[cand\] mode=pin" "$f" 2>/dev/null || true)
        echo "[ab]   preflight $arm: stderr=$(basename "$f")  mode=unified 行=$n_uni  mode=pin 行=$n_pin"
        if [ "$arm" = "A" ]; then
            [ "${n_uni:-0}" -ge 1 ] 2>/dev/null || { echo "[ab] !! A 臂没有 [cand] mode=unified ⇒ 会退化成 pin，停" >&2; ok=1; }
        else
            [ "${n_uni:-0}" = "0" ] 2>/dev/null || { echo "[ab] !! B 臂竟然出现 mode=unified ⇒ B 不是 pin，停" >&2; ok=1; }
            [ "${n_pin:-0}" -ge 1 ] 2>/dev/null || echo "[ab]   ⚠️ B 臂没有 mode=pin 行（不影响 A/B，但 n_cand 读数会缺）"
        fi
    done
    return "$ok"
}
SEED_PRE="${SEED_PRE:-80}"
echo "[ab] 前置校验：各臂 1 局 × 6 回合（seed=$SEED_PRE）"
preflight
RC_PRE=$?
# 清掉继承的 AZAI_*（preflight 里 export 过）
for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
if [ "$RC_PRE" -ne 0 ]; then
    echo "[ab] ❌ 前置校验失败 ⇒ 不跑正式批次" >&2; exit 5
fi
echo "[ab] ✓ 前置校验通过：A 臂确实走 unified、B 臂确实是 pin"
if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
    echo "[ab] PREFLIGHT_ONLY=1 ⇒ 到此为止"; exit 0
fi

run_arm() {          # $1 = A|B
    local arm="$1" tag extra suf
    if [ "$arm" = "A" ]; then
        tag="$TAG_A"; extra=";AZAI_CAND_MODE=unified;AZAI_CAND_VERBOSE=1"; suf="_UNA"
    else
        tag="$TAG_B"; extra=";AZAI_CAND_VERBOSE=1"; suf="_UNB"
    fi
    local env0="$BASE$extra"
    mkdir -p "$OUTROOT/$tag"
    echo ""
    echo "=============================================================="
    echo "[ab] 开跑 $arm 臂：tag=$tag  $(date '+%H:%M:%S')"
    echo "=============================================================="
    # 逐臂：清继承 → export 本臂（**不依赖任何继承**）
    local v kv
    for v in $(env | sed -n 's/^\(AZAI_[A-Za-z_0-9]*\)=.*/\1/p'); do unset "$v"; done
    for kv in $(echo "$env0" | tr ';' '\n'); do [ -n "$kv" ] && export "$kv"; done
    unset kv
    export MVS_STDERR_SUFFIX=1
    export DSH_PROGRESS_SUFFIX="$suf"
    export DSH_PROGRESS_TAG="$tag"
    echo "[ab]   AZAI_CAND_MODE=${AZAI_CAND_MODE:-<未设=pin>}  AZAI_CAND_VERBOSE=${AZAI_CAND_VERBOSE:-0}"
    echo "[ab]   AZAI_SEARCH=$AZAI_SEARCH AZAI_K=$AZAI_K FLAT_ROLLOUTS=$AZAI_FLAT_ROLLOUTS FLAT_ROUNDS=$AZAI_FLAT_ROUNDS FLAT_SEED=$AZAI_FLAT_SEED"
    "$P" -u _tmp_ladder.py --tag="$tag" --jobs="$JOBS" \
        --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" \
        --retry-invalid "$RETRY" --final-sweep --watchdog-stall-min 90 \
        $SEEDS 2>&1 | tee -a "match_results/ladder_logs/$tag/_run.log"
    local rc=${PIPESTATUS[0]}
    echo "[ab] $arm 臂结束 rc=$rc  $(date '+%H:%M:%S')"
    return "$rc"
}

RC=0
for arm in $ARMS; do
    run_arm "$arm" || RC=1
done
echo ""
echo "=============================================================="
echo "[ab] 全部结束 $(date '+%Y-%m-%d %H:%M:%S')  rc=$RC"
echo "[ab] 逐臂读数："
echo "  bash code/test_match/_unified_cand_ab_report.sh    # 或直接跑 analyze 脚本"
echo "=============================================================="
exit "$RC"
