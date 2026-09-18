#!/usr/bin/env bash
# 独立复核：价值先验 +18.8pp 是本项目对 rule_v4 的第一次真实提升，必须两件事都做。
#
#   (A) 混杂对照（在**原 seed 0..63** 上，可与已测各臂逐局配对）
#       uniform：走完全一样的"枚举钉类合法格 + 可执行性过滤"，但权重全相同
#       ⇒ 若 uniform 也明显强于位置网先验，那增益就来自"过滤掉不可执行格"而不是价值信息。
#
#   (B) 换 seed（64..127）重测三个关键臂 ⇒ 与 (A)/既有结果互相独立。
#       预期：位置网先验 ~20%，价值先验 ~35-40%，且 k=8 与 k=24 不可分。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt

run() {
  local name="$1"; local seed="$2"; shift 2
  echo "########## $name（seed=$seed） ##########"
  "$PY" -u code/my_ai/az_intent/eval.py \
      --checkpoint "$CK" --opponent rule_v4 --games 64 --workers 16 --seed "$seed" \
      --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 \
      --t-class 0.5 --search-mode pos-only --skip-single-candidate "$@"
}

run "A_uniform_tp1.0_k24（混杂对照）" 0  --pos-prior uniform --t-pos 1.0 --k 24
run "B_pol_tp1.0_k24（新 seed 基线）" 64 --pos-prior policy  --t-pos 1.0 --k 24
run "C_val_tp0.5_k24"                64 --pos-prior value   --t-pos 0.5 --k 24
run "D_val_tp0.5_k8"                 64 --pos-prior value   --t-pos 0.5 --k 8
