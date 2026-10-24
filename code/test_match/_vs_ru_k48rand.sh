#!/usr/bin/env bash
# 对外验证：#9 配置（k=48 + f10 + 随机化 N=64）打亚军 16 局
# 判据：血量差 vs 已知冠军基线 -26.3（若 >= -20 视为传导，>= -26.3 视为不掉）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
RUNNERUP="其他版本ai/saiblo-30th-AI/Game1/antgame_ai_cpp/cpp_lure_v4/build/ai_cpp_lure_v4.exe"
export AZAI_CKPT=checkpoints/prior_distill_f10.pt
export AZAI_ITERS=256 AZAI_DEPTH=4 AZAI_SAMPLE_MULT=15
export AZAI_MODE=pos-only AZAI_POSPIN=mc_quiet
export AZAI_MC_EVERY=1 AZAI_MC_ROOTONLY=1 AZAI_MC_CELLS=4
export AZAI_MC_HORIZON=16 AZAI_MC_SCORE=net AZAI_MC_NET_SAMPLES=4
export AZAI_SKIP1=1 AZAI_TCLASS=0.5 AZAI_TPOS=1.0 AZAI_TEMP=1e-6
export AZAI_POSPRIOR=off AZAI_SEARCH=flat
export AZAI_K=48
export AZAI_FLAT_ROUNDS=49920
export AZAI_FLAT_ROLLOUT_MODE=random AZAI_FLAT_RANDOM_SIDE=opponent
export AZAI_FLAT_PROB_MODE=decay AZAI_FLAT_DECAY_C=0.3 AZAI_FLAT_DECAY_K=1.0
export AZAI_FLAT_ACT_PROB=0.3 AZAI_FLAT_SHARE_PREFIX=1
export AZAI_FLAT_ROLLOUTS=64 AZAI_FLAT_SEED=41
echo "############ 对外: #9配置(k48+f10+随机N64) vs 亚军  16 局  $(date +%H:%M:%S)"
$PY -u _tmp_ladder.py --tag="VS_RU_k48rand" --jobs=8 \
  --ai0=code/test_match/az_bridge_ai.py --ai1="$RUNNERUP" \
  7 7r 8 8r 9 9r 10 10r 11 11r 12 12r 13 13r 14 14r
echo ""
echo "对外验证结束 ($(date +%H:%M:%S))"