#!/usr/bin/env bash
# 随机化强度 A/B（每局独立 FLAT_SEED）：B(强随机) vs 基线(弱随机)
#   B     : k=48, DECAY_C=1.0, DECAY_K=3.0, HORIZON=8   （实测有效叶子 62/64，约 5.5 s）
#   基线  : k=48, DECAY_C=0.3, DECAY_K=1.0, HORIZON=16  （实测有效叶子 17/64，约 4.8 s）
# 每局 FLAT_SEED = 1000 + seed（避免"64 局其实是同一条随机流"）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROUNDS=49920;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64"

echo "############ 随机化强度 A/B：B(强) vs 基线(弱)  每局独立 FLAT_SEED  $(date +%H:%M:%S)"
for seed in $(seq 3100 3115); do
  fs=$((1000 + seed))
  STRONG="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_HORIZON=8;AZAI_FLAT_SEED=$fs"
  WEAK="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SEED=$fs"
  "$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds "$seed" --jobs 2 \
    --out-dir "match_results/rollout_strength/seed${seed}" \
    --flat-env="$STRONG" --mcts-env="$WEAK" 2>&1 | tail -2
done
echo "结束 $(date +%H:%M:%S)"
