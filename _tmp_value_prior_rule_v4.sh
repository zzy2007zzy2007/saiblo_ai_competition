#!/usr/bin/env bash
# 在**强化对手 rule_v4** 上量三件事（镜像台在 2.0 处封顶，分辨不出这些差别）：
#   1) 缩 k 的代价：位置网先验 k=8 vs 已测的 k=24（20.3%）
#   2) **要追赶的上界**：价值网先验 k=8 —— "先验够好时 k 能缩多小而不掉强度"
#   3) 价值网先验 k=24（看它在强对手下是不是也更好）
# 全部同 seed=0 / 64 局，可与 tpos_rule_v4（位置网先验 tp1.0 k24 = 20.3%）逐局配对。
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt

run() {
  echo "########## $1 ##########"; shift
  "$PY" -u code/my_ai/az_intent/eval.py \
      --checkpoint "$CK" --opponent rule_v4 --games 64 --workers 16 --seed 0 \
      --bundle-mcts --native-engine --iterations 256 --max-depth-rounds 4 \
      --t-class 0.5 --search-mode pos-only --skip-single-candidate "$@"
}

run "A_pol_tp1.0_k8（缩 k 的代价）"   --pos-prior policy --t-pos 1.0 --k 8
run "B_val_tp0.5_k24（价值先验 k24）" --pos-prior value  --t-pos 0.5 --k 24
run "B_val_tp0.5_k8（要追赶的上界）"  --pos-prior value  --t-pos 0.5 --k 8
