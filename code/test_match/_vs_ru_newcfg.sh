#!/usr/bin/env bash
# **新配方打亚军**（绝对尺子）
#
# 新配方（今天两次内部实验验证的最佳）：
#   值网评分(MCCORE=net, 4 采样, h64) + 不升级 + cells4 + flat rollout
#   · 内部：net 71.9% 胜 quiet（都开升级@h64）；net@h64 69.2% 胜 quiet@H256（都不升级）
#   · 升级已确认有害三次，故关闭
#
# 目的：回答"值网评分把绝对水平抬了多少"。
#   今天的绝对基准：cells4+every1+H256 打亚军 = **1W-31L = 3.1%**（32 局）。
#   本臂用 8 张地图 × 2 座位 = **16 局**（配对），约 2 小时（亚军对局 ~60 min/局，8 并行）。
#
# 判读：
#   · 若胜率明显 > 3%（比如 ≥25%）⇒ 值网评分确实抬了绝对水平；
#   · 若仍是 0-1 胜 ⇒ 内部几十 pp 的提升不足以跨过与亚军的差距，得换思路（经济/多op/重训）。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r"

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_K=24 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_ROOTONLY=1 AZAI_MC_CELLS=4
export AZAI_MC_HORIZON=64 AZAI_MC_SCORE=net AZAI_MC_NET_SAMPLES=4
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat AZAI_FLAT_ROLLOUTS=16 AZAI_FLAT_HORIZON=64
export AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

echo "############ 新配方(net@h64,不升级) vs 亚军  16 局（8 图配对）  $(date +%H:%M:%S)"
$PY -u _tmp_ladder.py --tag="VS_RU_newcfg" --jobs=8 \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
    $SEEDS
echo ""
echo "新配方打亚军结束 ($(date +%H:%M:%S))"
