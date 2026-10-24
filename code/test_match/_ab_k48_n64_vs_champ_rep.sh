#!/usr/bin/env bash
# #9 复测（防小样本）：同配置换新种子，与既有 19W-13L 合并到 n=64
#   A = k=48 + f10 先验 + 随机化 N=64（预算 49920）
#   B = 冠军 k=160 + 确定性 N=1
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
B="AZAI_CKPT=checkpoints/prior_distill_f10.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_FLAT_HORIZON=16;AZAI_FLAT_BATCH=32;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4"
A="$B;AZAI_K=48;AZAI_FLAT_ROUNDS=49920;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=0.3;AZAI_FLAT_DECAY_K=1.0;AZAI_FLAT_ACT_PROB=0.3;AZAI_FLAT_SHARE_PREFIX=1;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_SEED=31"
CH="$B;AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_K=160;AZAI_FLAT_ROUNDS=39000;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_ROLLOUTS=1"
echo "############ #9 复测: k48+随机化N64(A) vs 冠军(B)  32 局配对  $(date +%H:%M:%S)"
$PY -u code/test_match/ab_search_modes.py --paired-sides --seeds $(seq 2540 2555 | tr '\n' ' ') --jobs 8 \
  --out-dir training_history/ab_k48_n64_vs_champ_rep \
  --flat-env="$A" --mcts-env="$CH"
echo ""
echo "#9复测 结束 ($(date +%H:%M:%S))"