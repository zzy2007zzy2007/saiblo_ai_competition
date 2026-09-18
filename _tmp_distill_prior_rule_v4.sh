#!/usr/bin/env bash
# 用"蒸馏版位置图"当先验，看能不能在**锐化（低 t_pos）**下保住强度。
#
# 背景（用户 2026-09-18 的假设）：现在的动作网是被遗传/BC 训坏的——遗传里"动作类选得好"
# 提升就大，位置选烂影响不大；而 `distill_data*/model.pt` 是直接拟合 SDK 启发式评分的产物，
# 每个合法操作都有监督 ⇒ 它的位置图可能反而相对好。若换成它之后锐化不再掉强度，
# 就说明"训练一个好位置先验"是有效的，可以走"小 k + 确定候选集"那条便宜路线。
#
# 对照（同 seed=0 / 64 局，可与本次逐局配对，均在 tpos_rule_v4 / tpos_rule_v4_mild 里）：
#   原先验 t_pos=1.0 -> 20.3%    原先验 t_pos=0.3 -> 1.6%   ← "锐化就崩"的参照
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
FIX=training_history/az_fixed
for spec in "V1_tp1.0:$FIX/three_mix_pos_distillV1.pt:1.0" \
            "V1_tp0.3:$FIX/three_mix_pos_distillV1.pt:0.3" \
            "V2_tp1.0:$FIX/three_mix_pos_distillV2.pt:1.0" \
            "V2_tp0.3:$FIX/three_mix_pos_distillV2.pt:0.3"; do
  name="${spec%%:*}"; rest="${spec#*:}"; ck="${rest%%:*}"; tp="${rest##*:}"
  echo "########## $name  (pos_prior=$(basename "$ck"), t_pos=$tp) ##########"
  "$PY" -u code/my_ai/az_intent/eval.py \
      --checkpoint "$ck" --opponent rule_v4 --games 64 --workers 16 --seed 0 \
      --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 \
      --k 24 --t-class 0.5 --t-pos "$tp" \
      --search-mode pos-only --skip-single-candidate
done
