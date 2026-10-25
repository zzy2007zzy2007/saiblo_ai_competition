#!/usr/bin/env bash
# 分散度假说检验：冠军网(posnet_A，尖) + 拉平采样(TEMP=5.0) @ k=48 + 随机化 N=64
#   A = k=48 + posnet_A + TEMP=5.0（拉平）+ 随机化 N=64
#   B = 冠军：k=160 + posnet_A + TEMP=1e-6（尖）+ 确定性 N=1
# 判据：A >= 55% ⇒ "候选分散度"是主因（与网无关）；A ≈ 40% ⇒ 仍是 f10 这张网特殊
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$B;AZAI_K=48;AZAI_FLAT_ROUNDS=49920;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=5.0;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=43"
CH="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$B;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
echo "############ 分散度检验: k48+posnetA+拉平(TEMP5.0)+随机化N64 vs 冠军  32 局  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2620 2635 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k48_flat_sampling \
  --flat-env="$A" --mcts-env="$CH"
echo ""
echo "结束 ($(date +%H:%M:%S))"