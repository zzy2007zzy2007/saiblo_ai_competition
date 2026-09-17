#!/usr/bin/env bash
# 补测"轻度尖锐"区间：t_pos ∈ (0.3, 1.0) 打 rule_v4。
# 动机：z-score 修复后 t_pos 的语义换了（旧值 ×381 ≈ 新值），所以 0.3 是一个
# **从未真正用过**的极尖档位（有效候选 10 / ~270 合法格，top1=0.47）。
# 已知 0.3 -> 1.6%（1W/63L）、1.0 -> 20.3%（13W/51L）。这里填中间：
#   t_pos=0.5 -> 先验有效候选 48、top1 0.21（自然的"轻度尖锐"）
#   t_pos=0.7 -> 中间点
# 其余配置与 _tmp_tpos_rule_v4.sh 完全一致（同 seed ⇒ 可与那三臂逐局配对）。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
for TP in 0.5 0.7; do
  echo "########## t_pos=$TP vs rule_v4 (64 games) ##########"
  "$PY" -u code/my_ai/az_intent/eval.py \
      --checkpoint "$CK" --opponent rule_v4 --games 64 --workers 16 --seed 0 \
      --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 \
      --k 24 --t-class 0.5 --t-pos "$TP" \
      --search-mode pos-only --skip-single-candidate
done
