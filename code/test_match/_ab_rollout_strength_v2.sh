#!/usr/bin/env bash
# 随机化强度 A/B v2：4 个 seed 并发（8 局同时），每局独立 FLAT_SEED
#   强随机: k=48, C=1.0, K=3.0, h=8   （有效叶子 62/64，约 5.5 s/决策）
#   基线  : k=48, C=0.3, K=1.0, h=16  （有效叶子 17/64，约 4.8 s/决策）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROUNDS=49920;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64"

echo "############ 随机化强度 A/B v2（4 seed 并发）  $(date +%H:%M:%S)"
for group in 0 1 2 3; do
  pids=""
  for j in 0 1 2 3; do
    seed=$((3100 + group * 4 + j))
    fs=$((1000 + seed))
    STRONG="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_HORIZON=8;AZAI_FLAT_SEED=$fs"
    WEAK="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SEED=$fs"
    (
      "$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds "$seed" --jobs 2 \
        --out-dir "match_results/rollout_strength/seed${seed}" \
        --flat-env="$STRONG" --mcts-env="$WEAK" 2>&1 | tail -3
    ) &
    pids="$pids $!"
  done
  echo "--- 第 $((group+1))/4 组启动（seeds $((3100 + group * 4))..$((3103 + group * 4))）  $(date +%H:%M:%S)"
  wait $pids
  echo "--- 第 $((group+1))/4 组完成  $(date +%H:%M:%S)"
done
echo "全部结束 $(date +%H:%M:%S)"
