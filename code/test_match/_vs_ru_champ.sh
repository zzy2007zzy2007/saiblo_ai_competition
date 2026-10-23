#!/usr/bin/env bash
# **新冠军打亚军**（绝对尺子，2026-10-23）
#
# 新冠军配方（今天由两次内部实验确定）：
#   · 值网仲裁 @h16（net 71.9% 胜 quiet；net@h64 69.2% 胜 quiet@H256）
#   · 不升级（三次实验一致为负：quiet 23.4%/18.8%，net 31.2%）
#   · **k=160 / 采样1 / h16**（k=160 曾 65.6% 胜部署口径；k=320 只 53.1% ⇒ 160 是甜点）
#
# 对照：今天早些时候同一批种子的绝对读数 —— cells4+every1+H256 = **1W-31L = 3.1%**；
#       新配方早期版（k=24/s16/h64/net）= **1W-15L = 6.2%**。
# 本臂：8 图 × 2 座位 = 16 局，jobs=8。
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
SEEDS="7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r"

export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_ROOTONLY=1 AZAI_MC_CELLS=4
export AZAI_MC_HORIZON=16 AZAI_MC_SCORE=net AZAI_MC_NET_SAMPLES=4
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_TRACE=1 AZAI_VERIFY=1
export AZAI_SEARCH=flat
export AZAI_K=160
export AZAI_FLAT_ROLLOUTS=1 AZAI_FLAT_HORIZON=16
export AZAI_FLAT_BATCH=32 AZAI_FLAT_ROUNDS=39000

echo "############ 新冠军(k=160/s1/h16/net/不升级) vs 亚军  16 局  $(date +%H:%M:%S)"
$PY -u _tmp_ladder.py --tag="VS_RU_champ" --jobs=8 \
    --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
    $SEEDS
echo ""
echo "新冠军打亚军结束 ($(date +%H:%M:%S))"
