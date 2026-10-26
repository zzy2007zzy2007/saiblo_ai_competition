#!/usr/bin/env bash
# 三方循环赛（每对 64 局）：f10 / prior_z_80g.best / hybrid
#   关键：**三组用完全相同的地图种子与 FLAT_SEED**，唯一变量是 ckpt
#   地图：2900-2931（--paired-sides ⇒ 每种子 2 局 = 64 局/对）
#   FLAT_SEED=61（随机化 rollout 的随机流固定）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
COMMON="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_K=48;AZAI_FLAT_ROUNDS=49920;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=61"
F10="AZAI_CKPT=checkpoints/prior_distill_f10.pt;$COMMON"
PZ="AZAI_CKPT=checkpoints/prior_z_80g.best.pt;$COMMON"
HY="AZAI_CKPT=checkpoints/prior_z_hybrid_light.pt;$COMMON"
SEEDS=$(seq 2900 2931 | tr '\n' ' ')

echo "############ 循环赛 1/3: f10 vs hybrid  $(date +%H:%M:%S)"
"$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 10 \
  --out-dir training_history/rr_f10_vs_hybrid --flat-env="$F10" --mcts-env="$HY"

echo "############ 循环赛 2/3: f10 vs prior_z  $(date +%H:%M:%S)"
"$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 10 \
  --out-dir training_history/rr_f10_vs_pz --flat-env="$F10" --mcts-env="$PZ"

echo "############ 循环赛 3/3: prior_z vs hybrid  $(date +%H:%M:%S)"
"$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds $SEEDS --jobs 10 \
  --out-dir training_history/rr_pz_vs_hybrid --flat-env="$PZ" --mcts-env="$HY"

echo "全部结束 $(date +%H:%M:%S)"
