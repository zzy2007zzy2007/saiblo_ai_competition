#!/usr/bin/env bash
# 端到端：pos-only 下"单候选跳过迭代"（--skip-single-candidate）开/关对比。
#
# 注意读法（见 docs/az_forced_move_skip_plan.md §二）：
#   * 本脚本**只比速度与冒烟，不比逐局结果**。跳过会改变 mcts.rng 之后抽到的数，
#     所以实现出来的对局本就会变——决策分布不变，具体实现不同。
#   * 等价性另有证据：同 RNG 状态下单次搜索逐位比对 660 次 0 差异（+120 次 temperature=1.0）。
set -u
REPO="D:/2026智能体大赛_新2"; PY="D:/anaconda3/envs/pytorch-gpu/python.exe"
cd "$REPO" || exit 1
CK=training_history/az_fixed/mix_r10p_vw_pol_frozen.pt
BASE="--checkpoint $CK --self-raw-opponent --bundle-mcts --native-engine \
      --iterations 256 --max-depth-rounds 4 --k 24 --t-class 0.5 --t-pos 0.3 \
      --games 8 --workers 8 --search-mode pos-only --pos-pin argmax"

run () {  # $1=标签 $2=额外 flag
  echo "########## $1 ##########"
  local s=$(date +%s)
  "$PY" code/my_ai/az_intent/eval.py $BASE $2
  echo ">>> $1 用时 $(( $(date +%s) - s ))s"
}

run "A: pos-only, skip 关（基线）" ""
run "B: pos-only + skip 开" "--skip-single-candidate"
echo "########## forced_move_skip 端到端完成 ##########"
