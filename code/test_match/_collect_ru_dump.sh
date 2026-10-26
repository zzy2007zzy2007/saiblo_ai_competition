#!/usr/bin/env bash
# 边打边 dump：冠军配置 vs 亚军（开启 AZAI_AZ_DUMP），用于离线算"我先验能否覆盖亚军走法"
#   6 局（3 图 × 2 座位），单局约 5-10 分钟
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
export AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_ROOTONLY=1 AZAI_MC_CELLS=4
export AZAI_MC_HORIZON=16 AZAI_MC_SCORE=net AZAI_MC_NET_SAMPLES=4
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_SEARCH=flat
export AZAI_K=160 AZAI_FLAT_ROUNDS=39000 AZAI_FLAT_ROLLOUT_MODE=idle AZAI_FLAT_ROLLOUTS=1
export AZAI_FLAT_HORIZON=16 AZAI_FLAT_BATCH=32
export AZAI_AZ_DUMP="D:/2026智能体大赛_新2_distill/training_history/az_dump_ru"
echo "############ 冠军 vs 亚军 + dump（6 局）  $(date +%H:%M:%S)"
"$PY" -u _tmp_ladder.py --tag="VS_RU_dump" --jobs=6 \
  --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
  7 7r 8 8r 9 9r
echo "结束 $(date +%H:%M:%S)"
