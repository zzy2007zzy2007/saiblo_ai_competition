#!/usr/bin/env bash
# 候选集来自官方启发式 top-K（ActionCatalog.build，含一步 rollout 重排），两侧在同一候选集里选。
#   control : 随机 vs 随机 ⇒ 应恰好 0.5000/SE=0
#   主判据   : 价值网 vs 随机 —— 价值网 ≫ 随机 ⇒ 它能排类动作 ⇒ 不用重训价值网
#   K=8/24  : K 越大价值网是否越排不动（winner's curse 的预测）
#   参照     : 纯启发式（first=ExampleAI 做法）vs rule_v4；价值网 vs rule_v4（绝对强度）
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
S=code/test_match/heuristic_candidates_match.py

echo "########## control: 随机 vs 随机 (k=8) ##########"
"$PY" -u $S --our random --opp random --k 8 --pairs 16 --workers 16 --seed 0
echo "########## 主判据: 价值网 vs 随机 (k=8) ##########"
"$PY" -u $S --our value --opp random --k 8 --pairs 32 --workers 16 --seed 0
echo "########## 价值网 vs 随机 (k=24) ##########"
"$PY" -u $S --our value --opp random --k 24 --pairs 32 --workers 16 --seed 0
echo "########## 参照: 纯启发式 vs rule_v4 (k=8) ##########"
"$PY" -u $S --our first --opp rule_v4 --k 8 --pairs 16 --workers 16 --seed 0
echo "########## 绝对强度: 价值网 vs rule_v4 (k=8) ##########"
"$PY" -u $S --our value --opp rule_v4 --k 8 --pairs 32 --workers 16 --seed 0
