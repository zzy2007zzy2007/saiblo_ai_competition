#!/usr/bin/env bash
# 把"尖度"和"内容质量"分开：同一条臂只换 t_pos，两条不同的先验各扫一遍。
#
#   oracle（价值偏好，暴力算，内容正确）× t_pos ∈ {0.3, 1.0, 2.0}
#     ⇒ 内容正确时"尖化"到底有没有害？（若基本是平的 ⇒ "尖有害"是内容差的性质）
#   learned（posnet_r1，部署用）× t_pos ∈ {0.3, 0.5, 2.0}
#     ⇒ 部署曲线（已有的 tp1.0 = 32.8%）
#
# 已知点（seed 0-63）：位置网先验 tp1.0 = 20.3%、tp0.3 = 1.6%；oracle tp0.5 = 39.1%。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
FIX=training_history/az_fixed
VP=training_history/vprior

run() {
  local name="$1"; local ck="$2"; shift 2
  echo "########## $name ##########"
  "$PY" -u code/my_ai/az_intent/eval.py --checkpoint "$ck" --opponent rule_v4 \
      --games 64 --workers 16 --seed 0 --bundle-mcts --native-engine --iterations 256 \
      --max-depth-rounds 4 --k 24 --t-class 0.5 --search-mode pos-only \
      --skip-single-candidate "$@"
}

for tp in 0.3 1.0 2.0; do
  run "oracle tp=$tp" "$FIX/three_mix_r10p_vw_pol_frozen.pt" --pos-prior value --t-pos "$tp"
done
for tp in 0.3 0.5 2.0; do
  run "learned tp=$tp" "$VP/posnet_r1.pt" --pos-prior policy --t-pos "$tp"
done
