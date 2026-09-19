#!/usr/bin/env bash
# rule_v4 的闪电位置：**只放闪电**的隔离检验（用户提议）。
# 只放闪电 = 冷却 0 且金币够就放闪电、否则空过（不建塔/升级/拆塔）⇒ 唯一的决策就是落点。
# 金币有被动收入（+1.5/轮），所以不放塔也能攒到 90。
#   control : 两侧都是"只放闪电 + rule_v4 的落点规则" ⇒ 应恰好 0.5000/SE=0（验证本模式对称性）
#   A       : 我方只放闪电（rule_v4 落点）vs **完整 rule_v4**
#   B/C/D   : 同上，但落点换成 我们的位置网 / 价值网 1-ply argmax / 256 迭代搜索
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
M="--mode lightning_only"

echo "########## control：两侧都是只放闪电（rule_v4 落点）##########"
"$PY" -u code/test_match/rule_v4_lightning_match.py $M --opp-mode lightning_only --pos-source none \
    --pairs 16 --workers 16 --seed 0

for spec in "A_rule_pos:none" "B_net:net" "C_value:value" "D_mcts:mcts"; do
  name="${spec%%:*}"; src="${spec#*:}"
  echo "########## $name（只放闪电，落点=$src）vs 完整 rule_v4 ##########"
  "$PY" -u code/test_match/rule_v4_lightning_match.py $M --opp-mode full --pos-source "$src" \
      --pairs 64 --workers 16 --seed 0
done
