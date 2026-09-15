#!/usr/bin/env bash
# 自对弈采集成本：现行配置 vs pos-only vs pos-only+单候选跳过。
#
# 现行生产配置（run_value_loop.sh:65）：256 / depth2 / --skip-hold-search / --write-npz
# 三臂逐项隔离：
#   A  现行        joint    + skip-hold
#   C  只换搜索轴  pos-only + skip-hold
#   B  再加单候选  pos-only + skip-hold + skip-single
#  => A vs C 量"换轴"的贡献，C vs B 量"单候选跳过"的贡献。
#
# ⚠️ pos-only 只适合采**价值头**数据（--write-npz 丢 bundles/visit，恰好只用锚点）。
#    若要采**策略**目标（visit 分布），pos-only 的 visit 是"固定类下的位置变体"，
#    与 joint 的分布不同，不要拿它训策略。
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1
CK=training_history/az_fixed/mix_r10p_vw_pol_frozen.pt
COMMON="--checkpoint $CK --games 16 --workers 16 --iterations 256 --max-rounds 512 \
        --native-engine --skip-hold-search --write-npz \
        --t-class 0.5 --t-pos 0.3 --k 24 --seed 900001"

run () {  # $1=标签 $2=out-dir 后缀 $3=额外 flag
  local out="training_history/_bench_collect/$2"
  rm -rf "$out"
  echo "########## $1 ##########"
  local s=$(date +%s)
  "$PY" code/my_ai/az_intent/az_selfplay.py $COMMON --max-depth-rounds 2 \
        --out-dir "$out" $3
  echo ">>> $1 用时 $(( $(date +%s) - s ))s"
}

run "A: 现行 joint + skip-hold"          a_joint  ""
run "C: pos-only + skip-hold"            c_posonly "--search-mode pos-only --pos-pin argmax"
run "B: pos-only + skip-hold + skip-single" b_both "--search-mode pos-only --pos-pin argmax --skip-single-candidate"
echo "########## 采集成本对照完成 ##########"
