#!/usr/bin/env bash
# 对手 = **亚军本体**，我们 = **flat + MC_EVERY=1 + MC_CELLS=4 + MC_HORIZON=128**。
#
# 与正在跑的 `VS_RUNNERUP_cells4`（同配置但 H=256）**用同一批 32 个种子**，
# 所以两者的差异可按 seed 配对看（同一张地图、同一先手）。
#
# 动机：
#   · 水平扫描：H=256 是离群点（vs rule_v4 0.891），H=128 有 0.984 ⇒ H256 可能纯属浪费
#   · H=256 + every1 时我们的 dec_max 已到 **7.53 s**（10 s 上限只剩 25% 余量）
#   · 砍到 128 预计把每决策的仲裁成本减半 ⇒ dec_max 显著下降
#
# 注意（统计）：两场各自 32 局，各自 2SE ≈ ±18pp ⇒ **只能粗比**；
# 若要精细判断"强度是否掉"，仍需内部对打。这里优先取"对真实对手的绝对成绩"。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r 15 15r 16 16r 17 17r 18 18r 19 19r 20 20r 21 21r 22 22r"

echo "等待 VS_RUNNERUP_cells4 结束… ($(date +%H:%M:%S))"
for _ in $(seq 1 500); do
  marker=$(ls -d training_history/runs/*vs_runnerup_cells4_v2* 2>/dev/null | tail -1)
  if [ -n "$marker" ] && grep -q "对亚军（cells4）测试结束" "$marker/output.log" 2>/dev/null; then
    echo "上一场已结束 ($(date +%H:%M:%S))"
    break
  fi
  sleep 30
done

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_HORIZON=128 AZAI_MC_ROOTONLY=1
export AZAI_MC_CELLS=4
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat
export AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64 AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

echo ""
echo "############ flat + every1 + CELLS4 + H128  vs 亚军 32 局  $(date +%H:%M:%S)"
$PY -u _tmp_ladder.py --tag="VS_RUNNERUP_cells4_h128" --jobs=8 \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
    $SEEDS
echo ""
echo "对亚军（cells4+H128）测试结束 ($(date +%H:%M:%S))"
