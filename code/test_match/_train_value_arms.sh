#!/usr/bin/env bash
# 值网重训 v1：在 `collect_v2_s1` 的 116 局上训 8 个标签臂（只换标签，其余全同）。
#
# 为什么另写：`train_value_net.py` 一次只训一个标签配置；这里把它按"臂表"跑一遍，
# 每臂一个 ckpt + 一个 log，方便后面统一跑 P0-2/P0-3 探针（筛子）与实战 A/B。
#
# 臂表（`--label-scale` 的定标规则：让标签 std ≈ terminal 的 0.531；
#       例外：`rel+geo τ50` 用计划里写死的 4.0）：
#   terminal        terminal              —     —   1.0
#   kgeo50 ★        abs  + kgeo  τ50  s1.5
#   kgeo25          abs  + kgeo  τ25  s1.5
#   kgeo10          abs  + kgeo  τ10  s1.5
#   kgeo100         abs  + kgeo  τ100 s1.5
#   rel50           rel  + geo   τ50  s4.0   ← 削捷径对照
#   relk50          rel  + kgeo  τ50  s3.4   ← 削捷径 + kgeo
#   abs50           abs  + geo   τ50  s1.5   ← 类型对照
#
# 用法： bash code/test_match/_train_value_arms.sh                # 全部 8 臂
#        ARMS="kgeo50 terminal" bash code/test_match/_train_value_arms.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export PYTHONIOENCODING=utf-8
P="${PYTHON:-D:/anaconda3/envs/pytorch-gpu/python.exe}"
CK="${CK:-checkpoints/posnet_A_k5_m32.pt}"
CACHE="${CACHE:-training_history/value_retrain_v1/cache_snap1}"
OUTD="${OUTD:-training_history/value_retrain_v1}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-3e-4}"
VAL_GAMES="${VAL_GAMES:-20}"
ARMS="${ARMS:-terminal kgeo50 kgeo25 kgeo10 kgeo100 rel50 relk50 abs50}"
mkdir -p "$OUTD/logs"

arm_spec() {
  case "$1" in
    terminal) echo "terminal geo 50 1.0" ;;
    kgeo50)   echo "abs kgeo 50 1.5" ;;
    kgeo25)   echo "abs kgeo 25 1.5" ;;
    kgeo10)   echo "abs kgeo 10 1.5" ;;
    kgeo100)  echo "abs kgeo 100 1.5" ;;
    rel50)    echo "rel geo 50 4.0" ;;
    relk50)   echo "rel kgeo 50 3.4" ;;
    abs50)    echo "abs geo 50 1.5" ;;
    *)        echo "" ;;
  esac
}

echo "=============================================================="
echo "[arms] cache=$CACHE  ckpt=$CK  epochs=$EPOCHS  lr=$LR  arms=[$ARMS]"
echo "[arms] $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================================="
fail=0
for arm in $ARMS; do
  spec="$(arm_spec "$arm")"
  if [ -z "$spec" ]; then echo "[arms] !! 未知臂 $arm" >&2; fail=1; continue; fi
  set -- $spec
  mode="$1"; weight="$2"; tau="$3"; scale="$4"
  out="$OUTD/arm_${arm}.pt"
  log="$OUTD/logs/${arm}.log"
  echo "[arm] $arm: mode=$mode weight=$weight tau=$tau scale=$scale -> $out"
  "$P" -u code/my_ai/az_intent/train_value_net.py \
      --ckpt "$CK" --cache "$CACHE" --epochs "$EPOCHS" --lr "$LR" \
      --label-mode "$mode" --label-weight "$weight" --tau "$tau" --label-scale "$scale" \
      --val-games "$VAL_GAMES" --out "$out" --device cuda > "$log" 2>&1
  rc=$?
  tail -2 "$log" | sed 's/^/    /'
  if [ "$rc" -ne 0 ]; then echo "[arm] !! $arm 失败 rc=$rc（见 $log）" >&2; fail=1; fi
done
echo "[arms] done（fail=$fail）"
exit "$fail"
