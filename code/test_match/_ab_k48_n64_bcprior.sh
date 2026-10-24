#!/usr/bin/env bash
# #9-b: 同配置但先验换成 bc_prior.pt（BC 版），检验收益是否依赖 f10
#   A = k=48 + bc_prior + 随机化 N=64
#   B = 冠军 k=160 + posnet_A + 确定性 N=1
# 判据：若 A 也显著（血量差 CI 不含 0 且 >=+3hp）⇒ 收益来自"先验质量"；若 A≈50% ⇒ 依赖 f10 特例
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A="AZAI_CKPT=checkpoints/bc_prior.pt;$B;AZAI_K=48;AZAI_FLAT_ROUNDS=49920;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=37"
CH="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;$B;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
echo "############ #9-b: k48+bc_prior+随机化N64(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2580 2595 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k48_n64_bcprior \
  --flat-env="$A" --mcts-env="$CH"
echo ""
echo "#9-b 结束 ($(date +%H:%M:%S))"