#!/usr/bin/env bash
# 干净对照 v2：双方都 idle，只差"是否消费引擎 RNG"（2026-10-28）
#   B（基准，固定）: idle + PERTURB=0（纯确定性 rollout，老部署配置）
#   A（被测）      : idle + PERTURB=1（唯一随机性来源 = 引擎 RNG 消费）
# 理由：实际部署只会二选一（动作随机 或 RNG 消费），不会叠加
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_PROB_MODE=const;AZAI_FLAT_ACT_PROB=0.5;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53"
ARM_B="$BASE;AZAI_FLAT_RNG_PERTURB=0"
ARM_A="$BASE;AZAI_FLAT_RNG_PERTURB=1"

echo "############ idle vs idle+扰动（唯一差别=RNG 消费）  $(date +%H:%M:%S)"
"$PY" -u code/test_match/ab_search_modes.py --paired-sides \
  --seeds $(seq 3500 3507 | tr '\n' ' ') --jobs 8 \
  --out-dir "match_results/rng_idle_clean" \
  --flat-env="$ARM_A" --mcts-env="$ARM_B" 2>&1 | tail -4
echo "结束 $(date +%H:%M:%S)（flat 胜率 = 被测臂 A（RNG 消费）的得分）"
