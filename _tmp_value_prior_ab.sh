#!/usr/bin/env bash
# 八臂对比（全部在**同一个台子** `_tmp_position_choice_ab.py`：镜像配对、对手=同一 ckpt 的
# raw 解码、同 seed ⇒ 逐局可比）。目的：回答三个此前靠跨台子数字猜的问题。
#
#   1) '搜索真的强于 1-ply 价值吗'（那条未结的口径差）
#        C_valargmax（不搜索，全部合法格取 1-ply 价值 argmax = pos_greedy 规则）
#        vs A_pol_tp1.0_k24（现有搜索）
#   2) '搜索的增益来自访问累积还是别的'
#        D_valpick（同一候选集，但直接按价值选，不看访问）vs A_pol_tp1.0_k24
#   3) '让价值网当先验（决定候选菜单）能不能缩 k'
#        B_val_*（根节点用价值 z-score 当采样分布）vs A_pol_*（位置网当先验）
#        并配 A_pol_tp1.0_k8 作为'缩 k'的对照臂
set -u
cd "$(dirname "$0")"
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
export PYTHONIOENCODING=utf-8
CK=training_history/az_fixed/three_mix_r10p_vw_pol_frozen.pt
COMMON="--checkpoint $CK --pairs 16 --workers 8 --seed 0"

run() { echo "########## $1 ##########"; shift; "$PY" -u _tmp_position_choice_ab.py $COMMON "$@"; }

run "A_pol_tp1.0_k24（基线）"        --mode search --pos-prior policy --t-pos 1.0 --k 24
run "A_pol_tp0.3_k24（尖先验）"      --mode search --pos-prior policy --t-pos 0.3 --k 24
run "A_pol_tp1.0_k8（缩 k）"         --mode search --pos-prior policy --t-pos 1.0 --k 8
run "B_val_tp0.5_k24"               --mode search --pos-prior value  --t-pos 0.5 --k 24
run "B_val_tp0.5_k8"                --mode search --pos-prior value  --t-pos 0.5 --k 8
run "B_val_tp1.0_k8"                --mode search --pos-prior value  --t-pos 1.0 --k 8
run "C_valargmax（不搜索）"           --mode valueargmax
run "D_valpick（同候选集按价值选）"    --mode valuepick --pos-prior policy --t-pos 1.0 --k 24
