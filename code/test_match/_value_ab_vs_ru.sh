#!/usr/bin/env bash
# 值网 A/B（实战）：同策略/类头/位置头，**只换 value_state**，对手 = 亚军 `ai_cpp_lure_v4.exe`。
#
# 为什么单写：现有 `_us_vs_external.sh` 依赖 `_tmp_ladder.py`（本机已删）⇒ 跑不起来；
# 这里用最小实现：`match_via_sdk.py` 每 token（seed / seedr）一局，xargs 并行。
#
# 用法：
#   bash code/test_match/_value_ab_vs_ru.sh A  checkpoints/posnet_A_k5_m32.pt   ""
#   bash code/test_match/_value_ab_vs_ru.sh B  training_history/value_retrain_v1/arm_relk50.pt ";AZAI_VALUE_TANH=0;AZAI_REL2ABS=3.4"
#   ARM=... TAG=... SEEDS="7 7r 8 8r" JOBS=4 bash code/test_match/_value_ab_vs_ru.sh ...
#
# 注意：
#  · 只覆盖 `AZAI_CKPT`（+ 可选的部署开关）；BASE 其余逐字取自 `_us_vs_external.sh:68`，与部署一致。
#  · 别把 JOBS 开太大：采集还在跑（12 核），这里默认 4，总数不超过 16 核。
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8

NAME="${1:-A}"
CKPT="${2:-checkpoints/posnet_A_k5_m32.pt}"
EXTRA="${3:-}"
TAG="${TAG:-VALUE_AB_${NAME}}"
JOBS="${JOBS:-4}"
SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r}"
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"

RU_REL="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
RU="$(pwd -W | sed 's#/$##')/$RU_REL"
BASE="$(sed -n 's/^BASE="\(.*\)"$/\1/p' code/test_match/_us_vs_external.sh)"
[ -n "$BASE" ] || { echo "[vab] !! 读不到 _us_vs_external.sh 的 BASE" >&2; exit 2; }
[ -f "$RU" ] || { echo "[vab] !! 亚军 exe 不存在：$RU" >&2; exit 2; }
for d in libstdc++-6.dll libgcc_s_seh-1.dll libwinpthread-1.dll; do
  [ -f "$(dirname "$RU")/$d" ] || { echo "[vab] !! 缺 $d（亚军会 0xC0000139 直接死）" >&2; exit 2; }
done

# 只替换 AZAI_CKPT；其余 BASE 键原样保留
BASE_NOCK="$(echo "$BASE" | tr ';' '\n' | grep -v '^AZAI_CKPT=' | paste -sd ';' -)"
ENV0="AZAI_CKPT=$CKPT;$BASE_NOCK$EXTRA"
LOGDIR="match_results/value_ab/$TAG"
mkdir -p "$LOGDIR/stderr"

echo "=============================================================="
echo "[vab] NAME=$NAME  TAG=$TAG  JOBS=$JOBS"
echo "[vab] CKPT=$CKPT"
echo "[vab] EXTRA=$EXTRA"
echo "[vab] 对手=亚军（$(basename "$RU")）  seeds=[$SEEDS]"
echo "[vab] env 摘要：$(echo "$ENV0" | tr ';' '\n' | grep -E 'AZAI_(CKPT|FLAT_|K=|CAND_|VALUE_|REL2ABS|MC_)' | paste -sd ';' -)"
echo "[vab] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="

run_one() {
  local tok="$1"
  local log="$LOGDIR/$tok.log"
  if [ -s "$log" ] && grep -q "^ *RESULT" "$log" 2>/dev/null; then
    echo "[vab] 跳过（已有 RESULT）：$tok"
    return 0
  fi
  echo "[vab] 开跑 $tok  $(date '+%H:%M:%S')"
  MVS_STDERR_SUFFIX=1 "$P" -u code/test_match/match_via_sdk.py \
      --rounds=512 \
      --ai0=code/test_match/az_bridge_ai.py --ai1="$RU" \
      --ai0-env="$ENV0" "$tok" > "$log" 2>&1
  local rc=$?
  echo "[vab] 完成 $tok rc=$rc  $(date '+%H:%M:%S')  $(grep -c '^ *RESULT' "$log" 2>/dev/null || echo 0) 条 RESULT"
}
export -f run_one 2>/dev/null || true
export LOGDIR P RU ENV0

printf '%s\n' $SEEDS | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
echo "[vab] 全部结束 $(date '+%Y-%m-%d %H:%M:%S')"
echo "[vab] 统计："
"$P" -u code/test_match/analyze_value_ab.py --dir "$LOGDIR" --name "$NAME" 2>&1 | tail -20
