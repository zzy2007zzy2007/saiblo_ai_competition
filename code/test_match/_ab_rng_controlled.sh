#!/usr/bin/env bash
# 随机性来源 A/B（控制变量版，2026-10-28）
#
# 设计（按用户要求）：
#   对手臂 A 完全固定：配置 + AZAI_FLAT_SEED=53 都不变，跨所有批次一致
#   实验臂 E 只变一个东西：AZAI_FLAT_SEED 逐批取 4300/4301/4302/4303
#   ⇒ 同一批次内，两组唯一差别是"随机性来源"
#   ⇒ 跨批次比较 E 的得分 ⇒ 直接测出 "FLAT_SEED 对 E（引擎RNG模式）的影响有多大"
#
#   A（固定）: ROLLOUT_MODE=random, PROB_MODE=decay C=1.0 K=3.0, ROUNDS=49152, RNG_PERTURB=0, FLAT_SEED=53
#   E（变种子）: ROLLOUT_MODE=idle, PROB_MODE=const p=0.5, ROUNDS=72253, RNG_PERTURB=1, FLAT_SEED=$ES
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64"

ARM_A="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=49152;AZAI_FLAT_SEED=53;AZAI_FLAT_RNG_PERTURB=0"

echo "############ 随机性来源 A/B（对手固定 FLAT_SEED=53）  $(date +%H:%M:%S)"
for ES in 4300 4301 4302 4303; do
  ARM_E="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_PROB_MODE=const;AZAI_FLAT_ACT_PROB=0.5;AZAI_FLAT_ROUNDS=72253;AZAI_FLAT_SEED=$ES;AZAI_FLAT_RNG_PERTURB=1"
  echo "=== 批次 FLAT_SEED(E)=$ES  地图 3300-3307  8 局×2 座位  $(date +%H:%M:%S)"
  "$PY" -u code/test_match/ab_search_modes.py --paired-sides \
    --seeds $(seq 3300 3307 | tr '\n' ' ') --jobs 8 \
    --out-dir "match_results/rng_ab_ctl/efs${ES}" \
    --flat-env="$ARM_E" --mcts-env="$ARM_A" 2>&1 | tail -3
  echo "=== 批次 $ES 完成  $(date +%H:%M:%S)"
done
echo "全部结束 $(date +%H:%M:%S)（读法：flat 胜率 = 实验臂 E 的得分）"
