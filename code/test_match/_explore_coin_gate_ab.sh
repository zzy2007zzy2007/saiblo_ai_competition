#!/usr/bin/env bash
# 自由探索 subagent 的"现役 × 金币门控"侦察 A/B（**只跑新臂**，对照臂复用
# `match_results/value_ab_rv4/AB_RV4_FULL/incumbent/` 的同 seeds 结果）。
#
# 动机（见 docs/free_exploration_strategy_arena_20261102.md §3/§7）：
#   · 竞技场（N=192/臂）说"边际金币给闪电、别花在塔/升级上"是 20 HP 量级的主轴；
#   · 2026-10-07 在**旧配置**上 `AZAI_BUILD_MIN_COINS=90` 拿到 +4.7pp、τ=60 +3.1pp（同号、剂量-反应），
#     但 64 局没过 0.7362 晋级线就被记档 ⇒ 值得在**今天的 unified 配置**上复现。
#   · 今天的配置金币更低（决策时刻中位数 31、p90=68 ⇒ 只有 3.9% 的回合能掏出 90），
#     所以 τ=90 可能把塔数打到 0；τ=60 是更保守的一档 ⇒ **两个都跑，看剂量-反应**。
#
# 用法（detached）：
#   bash code/test_match/_explore_coin_gate_ab.sh
#   JOBS=2 TOKENS="100 100r 101 101r" bash ...
#
# 产物：match_results/explore_20261102/coin_gate/<arm>/<token>.log + _times.tsv + _run.log
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-2}"
TOKENS="${TOKENS:-100 100r 101 101r 102 102r 103 103r}"
TAG="${TAG:-EXPLORE_COIN_GATE}"
OUTD="match_results/explore_20261102/coin_gate/$TAG"
RV4="code/test_match/rv4_pkg/main.py"
[ -f "$RV4" ] || { echo "[cg] !! 缺 $RV4" >&2; exit 2; }

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[cg] !! 读不到 BASE" >&2; exit 2; }
BASE_NOCK="$(echo "$BASE" | tr ';' '\n' | grep -v '^AZAI_CKPT=' | paste -sd ';' -)"

ARM_NAMES="${ARM_NAMES:-g60 g90}"
arm_extra() {
  case "$1" in
    g0)   echo "" ;;
    g45)  echo ";AZAI_BUILD_MIN_COINS=45" ;;
    g60)  echo ";AZAI_BUILD_MIN_COINS=60" ;;
    g90)  echo ";AZAI_BUILD_MIN_COINS=90" ;;
    noup) echo ";AZAI_BAN_OPS=12" ;;
    *)    echo "" ;;
  esac
}
CK="checkpoints/posnet_A_k5_m32.pt"

run_one() {
  local arm="$1" tok="$2"
  local dir log env0 t0 t1 rc
  dir="$OUTD/$arm"; log="$dir/$tok.log"; mkdir -p "$dir"
  if [ -s "$log" ] && grep -q "^ *RESULT" "$log" 2>/dev/null; then
    echo "[cg] skip $arm/$tok (已有 RESULT)"; return 0
  fi
  env0="AZAI_CAND_MODE=unified;AZAI_CKPT=$CK;$BASE_NOCK$(arm_extra "$arm")"
  export MVS_STDERR_SUFFIX=1
  export DSH_PROGRESS_SUFFIX="cg_${arm}_${tok}"
  t0=$(date +%s)
  "$P" -u code/test_match/match_via_sdk.py \
      --rounds=512 \
      --ai0=code/test_match/az_bridge_ai.py --ai1="$RV4" \
      --ai0-env="$env0" "$tok" > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\n" "$arm" "$tok" "$((t1 - t0))" "$rc" >> "$OUTD/_times.tsv"
  echo "[cg] $arm/$tok rc=$rc $(( (t1 - t0) / 60 )) min $(date '+%H:%M:%S')"
}
export -f run_one arm_extra
export P RV4 BASE_NOCK OUTD CK

mkdir -p "$OUTD"
{
  echo "=============================================================="
  echo "[cg] TAG=$TAG JOBS=$JOBS 臂=[$ARM_NAMES] tokens=[$TOKENS]"
  echo "[cg] 对手=rule_v4  BASE=$(echo "$BASE_NOCK" | cut -c1-80)…"
  echo "[cg] $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================================================="
} | tee -a "$OUTD/_run.log"

JOBS_LIST=""
for tok in $TOKENS; do
  for arm in $ARM_NAMES; do
    JOBS_LIST="$JOBS_LIST$arm|$tok
"
  done
done
echo -n "$JOBS_LIST" | sed '/^$/d' | xargs -P "$JOBS" -I{} bash -c '
  IFS="|" read -r a t <<< "{}"; run_one "$a" "$t"' 2>&1 | tee -a "$OUTD/_run.log"

echo "[cg] 全部结束 $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$OUTD/_run.log"
