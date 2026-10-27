#!/usr/bin/env bash
# 随机性来源 A/B（2026-10-28 明早跑）：动作随机 vs 引擎 RNG 消费
#
#  对照 A : ROLLOUT_MODE=random, PROB_MODE=decay, C=1.0, K=3.0, RNG_PERTURB=0
#           实测 62.4/64 叶子、938 节点/候选、11.15 s
#  实验 E : ROLLOUT_MODE=idle,   PROB_MODE=const, ACT_PROB=0.5, RNG_PERTURB=1
#           实测 64.0/64 叶子、711 节点/候选、7.63 s（A 的 68%）
#
# 等墙钟对照：E 的回合数乘 1.47（7.63/11.15 的倒数），否则 E 只是"跑得快的同一件事"
# 每局独立 FLAT_SEED = 1000 + seed（避免"64 局其实是同一条随机流"）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=D:/anaconda3/envs/pytorch-gpu/python.exe
BASE="AZAI_CKPT=checkpoints/posnet_A_k5_m32.pt;AZAI_ITERS=256;AZAI_DEPTH=4;AZAI_SAMPLE_MULT=15;AZAI_MODE=pos-only;AZAI_POSPIN=mc_quiet;AZAI_MC_EVERY=1;AZAI_MC_ROOTONLY=1;AZAI_MC_CELLS=4;AZAI_SKIP1=1;AZAI_TCLASS=0.5;AZAI_TPOS=1.0;AZAI_TEMP=1e-6;AZAI_POSPRIOR=off;AZAI_SEARCH=flat;AZAI_MC_HORIZON=16;AZAI_MC_SCORE=net;AZAI_MC_NET_SAMPLES=4;AZAI_FLAT_BATCH=32;AZAI_FLAT_RANDOM_SIDE=opponent;AZAI_FLAT_HORIZON=16;AZAI_FLAT_SHARE_PREFIX=1;AZAI_K=48;AZAI_FLAT_ROLLOUTS=64"

R_A=49152
R_E=$((49152 * 147 / 100))   # 72200：按 1.47 倍补偿 E 的更快墙钟

echo "############ 随机性来源 A/B（动作随机 vs 引擎RNG）  $(date +%H:%M:%S)"
echo "  对照 ROUNDS=$R_A   实验 ROUNDS=$R_E"
for group in 0 1 2 3; do
  pids=""
  for j in 0 1 2 3; do
    seed=$((3300 + group * 4 + j))
    fs=$((1000 + seed))
    ARM_A="$BASE;AZAI_FLAT_ROLLOUT_MODE=random;AZAI_FLAT_PROB_MODE=decay;AZAI_FLAT_DECAY_C=1.0;AZAI_FLAT_DECAY_K=3.0;AZAI_FLAT_ROUNDS=$R_A;AZAI_FLAT_SEED=$fs;AZAI_FLAT_RNG_PERTURB=0"
    ARM_E="$BASE;AZAI_FLAT_ROLLOUT_MODE=idle;AZAI_FLAT_PROB_MODE=const;AZAI_FLAT_ACT_PROB=0.5;AZAI_FLAT_ROUNDS=$R_E;AZAI_FLAT_SEED=$fs;AZAI_FLAT_RNG_PERTURB=1"
    (
      "$PY" -u code/test_match/ab_search_modes.py --paired-sides --seeds "$seed" --jobs 2 \
        --out-dir "match_results/rng_perturb_ab/seed${seed}" \
        --flat-env="$ARM_E" --mcts-env="$ARM_A" 2>&1 | tail -3
    ) &
    pids="$pids $!"
  done
  echo "--- 第 $((group+1))/4 组启动（seeds $((3300 + group * 4))..$((3303 + group * 4))）  $(date +%H:%M:%S)"
  wait $pids
  echo "--- 第 $((group+1))/4 组完成  $(date +%H:%M:%S)"
done
echo "全部结束 $(date +%H:%M:%S)（读法：flat 胜率 = 实验组 E 的得分）"
