#!/usr/bin/env bash
# rule_v4 闪电位置迁移性检验（两侧都是 rule_v4 血统、镜像配对、确定性）。
#   NULL  : 我方也用原版 ⇒ 应为恰好 0.5000 / SE=0（零方差对照）
#   net   : 我方闪电位置 = 我们训好的位置网 posnet_r1 的 argmax（1 次前向，可部署形态）
#   value : 位置 = 价值网逐格 1-ply 评估取最优（约 270 次前向，oracle 形态）
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
for spec in "null:--null" "net:--pos-source net" "value:--pos-source value"; do
  name="${spec%%:*}"; opt="${spec#*:}"
  echo "########## $name ##########"
  "$PY" -u code/test_match/rule_v4_lightning_match.py $opt --pairs 64 --workers 16 --seed 0
done
