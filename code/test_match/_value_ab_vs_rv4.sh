#!/usr/bin/env bash
# 值网 A/B × 4 臂，对手 = **rule_v4**（Python 规则 AI，`code/test_match/rv4_pkg/main.py`）。
#
# 为什么换成 rule_v4：亚军太强，胜率天花板把值网差异埋掉；rule_v4 弱 ⇒ 臂间差异更可测，
# 同时顺带回答"统一搜索到底打不打得过 rule_v4"。
#
# 设计（**镜像配对 + 交错**）：
#   · 每个 seed 跑两个座位（`100` 我们执先手 / `100r` 我们执后手）⇒ 每对镜像抵消先手优势；
#   · 外层循环是 (seed, 座位)，内层是臂 ⇒ 早完成的批次里四臂都有数据（可提前看方向）。
#   · 臂只换 `AZAI_CKPT`（+ rel 臂的部署开关），搜索配置 = 部署 BASE（取自 `_us_vs_external.sh:68`）。
#
# 用法：
#   bash code/test_match/_value_ab_vs_rv4.sh                 # 默认 4 臂 × seeds 100..115 × 2 座位
#   JOBS=6 SEEDS="100 100r 101 101r" bash code/test_match/_value_ab_vs_rv4.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-$(for s in $(seq 100 115); do echo -n "$s ${s}r "; done)}"
TAG="${TAG:-AB_RV4}"
OUTD="match_results/value_ab_rv4/$TAG"
RV4="code/test_match/rv4_pkg/main.py"
[ -f "$RV4" ] || { echo "[ab4] !! 缺 $RV4" >&2; exit 2; }

BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[ab4] !! 读不到 BASE" >&2; exit 2; }
BASE_NOCK="$(echo "$BASE" | tr ';' '\n' | grep -v '^AZAI_CKPT=' | paste -sd ';' -)"

# 臂表：名字|ckpt|额外 env（rel 标签要 AZAI_VALUE_TANH=0 + AZAI_REL2ABS=scale）
ARM_NAMES="${ARM_NAMES:-incumbent kgeo50 abs50 relk50}"
arm_ckpt() {
  case "$1" in
    incumbent) echo "checkpoints/posnet_A_k5_m32.pt" ;;
    kgeo50)    echo "training_history/value_retrain_v1/arm_kgeo50.pt" ;;
    abs50)     echo "training_history/value_retrain_v1/arm_abs50.pt" ;;
    kgeo25)    echo "training_history/value_retrain_v1/arm_kgeo25.pt" ;;
    relk50)    echo "training_history/value_retrain_v1/arm_relk50.pt" ;;
    rel50)     echo "training_history/value_retrain_v1/arm_rel50.pt" ;;
    terminal)  echo "training_history/value_retrain_v1/arm_terminal.pt" ;;
    aux2)      echo "training_history/value_retrain_v1/arm_aux2_kgeo50.pt" ;;
    # 实验 B：只改候选游标深度（值网 = 现役）
    steps24|steps48|steps6) echo "checkpoints/posnet_A_k5_m32.pt" ;;
    *)         echo "" ;;
  esac
}
arm_extra() {
  case "$1" in
    relk50) echo ";AZAI_VALUE_TANH=0;AZAI_REL2ABS=3.4" ;;
    rel50)  echo ";AZAI_VALUE_TANH=0;AZAI_REL2ABS=4.0" ;;
    steps24) echo ";AZAI_CAND_STEPS=24" ;;
    steps48) echo ";AZAI_CAND_STEPS=48" ;;
    steps6)  echo ";AZAI_CAND_STEPS=6" ;;
    *)      echo "" ;;
  esac
}

run_one() {
  local arm="$1" tok="$2"
  local ck dir log env0
  ck="$(arm_ckpt "$arm")"
  dir="$OUTD/$arm"
  log="$dir/$tok.log"
  mkdir -p "$dir"
  if [ -s "$log" ] && grep -q "^ *RESULT" "$log" 2>/dev/null; then
    return 0
  fi
  # ⚠️ 必须显式给 `AZAI_CAND_MODE=unified`：BASE（`_us_vs_external.sh:68`）里**没有**这个键，
  #    缺了它 bridge 会退化成 pin+flat（老的 FlatRolloutGreedy），跑的不是"统一候选 + flat 阶段 2"。
  env0="AZAI_CAND_MODE=unified;AZAI_CKPT=$ck;$BASE_NOCK$(arm_extra "$arm")"
  # 每个进程的 stderr 单独命名（默认名只含 seat/label/seed ⇒ 4 臂同 seed 并发会互相覆盖，
  # 而 `[cand] mode=unified` 那行证据就在这个文件里）
  export MVS_STDERR_SUFFIX=1
  export DSH_PROGRESS_SUFFIX="${arm}_${tok}"
  local t0 t1
  t0=$(date +%s)
  "$P" -u code/test_match/match_via_sdk.py \
      --rounds=512 \
      --ai0=code/test_match/az_bridge_ai.py --ai1="$RV4" \
      --ai0-env="$env0" "$tok" > "$log" 2>&1
  local rc=$?
  t1=$(date +%s)
  # 每局墙钟（实验 B 要比 steps=12/24/48 的耗时）
  printf "%s\t%s\t%s\t%s\n" "$arm" "$tok" "$((t1 - t0))" "$rc" >> "$OUTD/_times.tsv"
  echo "[ab4] $arm/$tok rc=$rc $(( (t1 - t0) / 60 )) min $(date '+%H:%M:%S')"
}

export -f run_one arm_ckpt arm_extra
export P RV4 BASE_NOCK OUTD

mkdir -p "$OUTD"
echo "=============================================================="
echo "[ab4] TAG=$TAG JOBS=$JOBS"
echo "[ab4] 臂=[$ARM_NAMES]  对手=rule_v4（$RV4）"
echo "[ab4] seeds=[$SEEDS]"
echo "[ab4] 搜索=部署 BASE（$(echo "$BASE_NOCK" | cut -c1-60)…）"
echo "[ab4] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

# 交错展开：外层 (seed,座位)，内层臂 ⇒ 早期批次四臂都有数据
JOBS_LIST=""
for tok in $SEEDS; do
  for arm in $ARM_NAMES; do
    JOBS_LIST="$JOBS_LIST$arm|$tok
"
  done
done
echo -n "$JOBS_LIST" | sed '/^$/d' | xargs -P "$JOBS" -I{} bash -c '
  IFS="|" read -r a t <<< "{}"; run_one "$a" "$t"' 2>&1 | tee -a "$OUTD/_run.log"

echo "[ab4] 全部结束 $(date '+%Y-%m-%d %H:%M:%S')"
DIRS=""; NAMES=""
for arm in $ARM_NAMES; do DIRS="$DIRS $OUTD/$arm"; NAMES="$NAMES $arm"; done
# shellcheck disable=SC2086
"$P" -u code/test_match/analyze_value_ab.py --dir $DIRS --name $NAMES 2>&1 | tail -30
