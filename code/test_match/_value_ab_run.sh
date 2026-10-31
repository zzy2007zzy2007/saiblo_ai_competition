#!/usr/bin/env bash
# 一键跑「值网 A/B」两臂（A = 现役，B = 重训臂），按 SEEDS 配对座位。
#
# 用法： bash code/test_match/_value_ab_run.sh
#        ARM_B=arm_kgeo25.pt EXTRA_B=";" bash code/test_match/_value_ab_run.sh   # 换 B 臂
# 环境：JOBS（默认 4）、SEEDS（默认 seeds 7..10 × 两座位 = 8 局/臂）
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || exit 1
export JOBS="${JOBS:-4}"
export SEEDS="${SEEDS:-7 7r 8 8r 9 9r 10 10r}"
CK_A="${CK_A:-checkpoints/posnet_A_k5_m32.pt}"
CK_B="${CK_B:-training_history/value_retrain_v1/arm_relk50.pt}"
# B 臂是 rel 标签 ⇒ 部署必须把相对值还原成绝对值（`az_selfplay.py:187-188` 的 value/rel2abs + hp_delta/20）
EXTRA_B="${EXTRA_B-;AZAI_VALUE_TANH=0;AZAI_REL2ABS=3.4}"

echo "[ab] JOBS=$JOBS SEEDS=[$SEEDS]"
echo "[ab] A: $CK_A"
echo "[ab] B: $CK_B   extra='$EXTRA_B'"
bash code/test_match/_value_ab_vs_ru.sh A "$CK_A" "" || true
bash code/test_match/_value_ab_vs_ru.sh B "$CK_B" "$EXTRA_B" || true
echo "[ab] 两臂结束；汇总："
D:/anaconda3/envs/pytorch-gpu/python.exe -u code/test_match/analyze_value_ab.py \
    --dir match_results/value_ab/VALUE_AB_A match_results/value_ab/VALUE_AB_B \
    --name A_incumbent B_relk50 2>&1 | tail -25
