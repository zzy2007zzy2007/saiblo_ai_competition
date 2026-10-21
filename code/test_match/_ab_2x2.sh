#!/usr/bin/env bash
# 2×2 四臂：{mcts, flat} × {MC_EVERY=5, MC_EVERY=1}，全部 vs rule_v4（外部锚点）。
#
# 目的：
#   ① **验证** mcts+every5 这条臂是否复现已知基线 0.7344（若不复现 ⇒ 前面的 head-to-head 不成立）；
#   ② 测用户提出的「MC 仲裁从 20% 回合提到 100% 回合」（MC_EVERY=1）是否更强；
#   ③ 检验两者是否叠加（flat + every1）。
#
# 每臂 64 局（32 seed × 双向镜像），jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r 19 19r 20 20r 21 21r 22 22r 23 23r 24 24r 25 25r 26 26r 27 27r 28 28r 29 29r 30 30r 31 31r 32 32r 33 33r 34 34r 35 35r 36 36r 37 37r 38 38r"

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_HORIZON=256 AZAI_MC_ROOTONLY=1
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1

run_arm () {
  local tag="$1" search="$2" every="$3"
  export AZAI_SEARCH="$search"
  export AZAI_MC_EVERY="$every"
  echo ""
  echo "############################################################"
  echo "## ARM $tag : search=$search MC_EVERY=$every  ($(date +%H:%M:%S))"
  echo "############################################################"
  $PY -u _tmp_ladder.py --tag="$tag" --jobs=8 \
      --ai0=code/test_match/az_bridge_ai.py --ai1=code/test_match/rv4_pkg/main.py \
      $SEEDS
}

run_arm "A1_mcts_every5"  mcts 5
run_arm "A2_mcts_every1"  mcts 1
run_arm "A3_flat_every5"  flat 5
run_arm "A4_flat_every1"  flat 1
echo ""
echo "全部四臂结束 ($(date +%H:%M:%S))"
